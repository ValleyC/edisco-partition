"""
EDISCO-Partition: E(2)-Equivariant Partition Network for Large-Scale CVRP
=========================================================================

This module implements the complete EDISCO-Partition framework that combines:
1. E(2)-Equivariant partitioning for customer clustering
2. EDISCO diffusion-based routing for per-cluster optimization
3. End-to-end training with multiple loss components

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │              EDISCO-Partition Framework                      │
    ├─────────────────────────────────────────────────────────────┤
    │  Stage 1: EquivariantPartitionEncoder                       │
    │      - Shared EGNN backbone for geometric processing        │
    │      - Multi-scale feature extraction                       │
    │      - Capacity-aware cluster head                          │
    │                                                              │
    │  Stage 2: ClusterRouter (per-cluster)                       │
    │      - Reuses pretrained EDISCO or lightweight routing      │
    │      - Parallel processing of clusters                       │
    │                                                              │
    │  Stage 3: SolutionAssembler                                 │
    │      - Combines cluster routes into full solution           │
    │      - Optional inter-cluster refinement                     │
    └─────────────────────────────────────────────────────────────┘

Author: EDISCO Team
For: IEEE TPAMI Journal Extension
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import List, Dict, Tuple, Optional, Union
import math
from dataclasses import dataclass
from enum import Enum
import warnings


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class PartitionConfig:
    """Configuration for the partition network."""
    # Architecture
    n_layers: int = 8
    node_dim: int = 128
    hidden_dim: int = 256
    n_heads: int = 8
    dropout: float = 0.1
    coord_dim: int = 2

    # Partitioning
    n_clusters: int = 10
    min_cluster_size: int = 5
    max_cluster_size: int = 100

    # Training
    use_checkpoint: bool = False  # Gradient checkpointing for memory

    # Loss weights
    balance_weight: float = 1.0
    compactness_weight: float = 0.5
    entropy_weight: float = 0.1
    supervised_weight: float = 1.0
    coverage_weight: float = 0.5


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: float = 1.0) -> torch.Tensor:
    """Sinusoidal timestep embedding."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


# ============================================================================
# E(2)-EQUIVARIANT LAYERS
# ============================================================================

class EGNNLayer(nn.Module):
    """
    E(n)-Equivariant Graph Neural Network Layer.

    Maintains strict equivariance:
    - Coordinates transform equivariantly: f(Rx + t) = Rf(x) + t
    - Node/edge features remain invariant: h(Rx + t) = h(x)

    Key design principles:
    1. Messages computed from invariant quantities only (distances, features)
    2. Coordinate updates via scalar × vector operations
    3. No absolute position information leaks into features
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        coord_dim: int = 2,
        dropout: float = 0.1,
        use_attention: bool = True,
        n_heads: int = 4
    ):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.coord_dim = coord_dim
        self.use_attention = use_attention
        self.n_heads = n_heads

        # Message network - ONLY invariant inputs
        # Input: h_i || h_j || e_ij || d_ij (distance)
        message_input_dim = node_dim * 2 + edge_dim + 1
        self.message_net = nn.Sequential(
            nn.Linear(message_input_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Attention mechanism (optional)
        if use_attention:
            self.attention_net = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.SiLU(),
                nn.Linear(hidden_dim // 2, n_heads)
            )
            self.head_dim = hidden_dim // n_heads
            self.value_proj = nn.Linear(hidden_dim, hidden_dim)

        # Coordinate update network - outputs SCALAR weights
        self.coord_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1, bias=False)  # No bias for equivariance
        )

        # Node update network
        self.node_net = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, node_dim)
        )

        # Edge update network
        self.edge_net = nn.Sequential(
            nn.Linear(edge_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, edge_dim)
        )

        # Layer norms for residual connections
        self.node_norm = nn.LayerNorm(node_dim)
        self.edge_norm = nn.LayerNorm(edge_dim)

        # Coordinate update scaling (learnable)
        self.coord_scale = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        e: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with E(2) equivariance.

        Args:
            h: Node features (batch, n_nodes, node_dim) - INVARIANT
            x: Coordinates (batch, n_nodes, coord_dim) - EQUIVARIANT
            e: Edge features (batch, n_nodes, n_nodes, edge_dim) - INVARIANT
            mask: Optional attention mask (batch, n_nodes, n_nodes)

        Returns:
            h_new, x_new, e_new
        """
        batch_size, n_nodes, _ = h.shape

        # Compute pairwise quantities
        # x_diff is EQUIVARIANT: (Rx_j + t) - (Rx_i + t) = R(x_j - x_i)
        x_i = x.unsqueeze(2)  # (batch, n, 1, coord_dim)
        x_j = x.unsqueeze(1)  # (batch, 1, n, coord_dim)
        x_diff = x_j - x_i    # (batch, n, n, coord_dim)

        # Distance is INVARIANT: ||R(x_j - x_i)|| = ||x_j - x_i||
        distances = torch.norm(x_diff, dim=-1, keepdim=True)  # (batch, n, n, 1)

        # Prepare node features for message passing
        h_i = h.unsqueeze(2).expand(-1, -1, n_nodes, -1)  # (batch, n, n, node_dim)
        h_j = h.unsqueeze(1).expand(-1, n_nodes, -1, -1)  # (batch, n, n, node_dim)

        # Compute messages from INVARIANT quantities only
        msg_input = torch.cat([h_i, h_j, e, distances], dim=-1)
        messages = self.message_net(msg_input)  # INVARIANT: (batch, n, n, hidden_dim)

        # Apply mask if provided
        if mask is not None:
            messages = messages * mask.unsqueeze(-1)

        # Attention mechanism
        if self.use_attention:
            attn_logits = self.attention_net(messages)  # (batch, n, n, n_heads)
            attn_logits = attn_logits / math.sqrt(self.head_dim)

            if mask is not None:
                attn_logits = attn_logits.masked_fill(mask.unsqueeze(-1) == 0, -1e4)

            attn_weights = F.softmax(attn_logits, dim=2)  # (batch, n, n, n_heads)

            # Multi-head aggregation
            values = self.value_proj(messages)  # (batch, n, n, hidden_dim)
            values = values.view(batch_size, n_nodes, n_nodes, self.n_heads, self.head_dim)
            attn_weights = attn_weights.unsqueeze(-1)  # (batch, n, n, n_heads, 1)

            messages_attended = (attn_weights * values).sum(dim=2)  # (batch, n, n_heads, head_dim)
            messages_attended = messages_attended.view(batch_size, n_nodes, self.hidden_dim)
        else:
            messages_attended = messages.sum(dim=2)  # (batch, n, hidden_dim)

        # Update coordinates EQUIVARIANTLY
        # Scalar weights are INVARIANT
        coord_weights = self.coord_net(messages)  # (batch, n, n, 1)
        coord_weights = torch.tanh(coord_weights / 10.0)  # Stability

        # Normalized direction (EQUIVARIANT)
        x_diff_norm = x_diff / (distances + 1e-8)

        # Equivariant update: sum of (INVARIANT scalar × EQUIVARIANT vector)
        coord_update = (coord_weights * x_diff_norm).sum(dim=2)  # (batch, n, coord_dim)
        x_new = x + self.coord_scale * coord_update

        # Update node features (INVARIANT)
        h_input = torch.cat([h, messages_attended], dim=-1)
        h_new = self.node_norm(h + self.node_net(h_input))

        # Update edge features (INVARIANT)
        e_input = torch.cat([e, messages], dim=-1)
        e_new = self.edge_norm(e + self.edge_net(e_input))

        return h_new, x_new, e_new


class MultiScaleEGNN(nn.Module):
    """
    Multi-scale E(2)-Equivariant encoder with skip connections.

    Captures both local and global geometric patterns through:
    1. Progressive message passing at multiple scales
    2. Skip connections preserving fine-grained information
    3. Global attention for long-range dependencies
    """

    def __init__(self, config: PartitionConfig):
        super().__init__()
        self.config = config

        # Input projections
        self.node_embed = nn.Sequential(
            nn.Linear(2, config.node_dim),  # [demand/capacity, is_depot]
            nn.LayerNorm(config.node_dim),
            nn.SiLU(),
            nn.Linear(config.node_dim, config.node_dim)
        )

        self.edge_embed = nn.Sequential(
            nn.Linear(1, config.hidden_dim),  # Distance
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )

        # EGNN layers with skip connections
        self.layers = nn.ModuleList()
        self.skip_projs = nn.ModuleList()

        for i in range(config.n_layers):
            self.layers.append(
                EGNNLayer(
                    node_dim=config.node_dim,
                    edge_dim=config.hidden_dim,
                    hidden_dim=config.hidden_dim,
                    coord_dim=config.coord_dim,
                    dropout=config.dropout,
                    use_attention=(i >= config.n_layers // 2),  # Attention in later layers
                    n_heads=config.n_heads
                )
            )

            # Skip connection projection (every 2 layers)
            if (i + 1) % 2 == 0 and i < config.n_layers - 1:
                self.skip_projs.append(
                    nn.Linear(config.node_dim * 2, config.node_dim)
                )

        # Global context aggregation
        self.global_attention = nn.MultiheadAttention(
            embed_dim=config.node_dim,
            num_heads=config.n_heads,
            dropout=config.dropout,
            batch_first=True
        )

        self.output_norm = nn.LayerNorm(config.node_dim)

    def forward(
        self,
        coords: torch.Tensor,
        invariant_features: torch.Tensor,
        return_intermediates: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Multi-scale equivariant encoding.

        Args:
            coords: (batch, n_nodes, 2) - coordinates
            invariant_features: (batch, n_nodes, 2) - [demand/capacity, is_depot]
            return_intermediates: Whether to return intermediate representations

        Returns:
            h: Final node embeddings (invariant)
            [optional] intermediates: List of intermediate embeddings
        """
        batch_size, n_nodes, _ = coords.shape

        # Embed invariant features
        h = self.node_embed(invariant_features)

        # Compute pairwise distances for edge features
        dist_matrix = torch.cdist(coords, coords)
        e = self.edge_embed(dist_matrix.unsqueeze(-1))

        # Keep coordinates for equivariant processing
        x = coords.clone()

        intermediates = [h] if return_intermediates else None
        skip_idx = 0
        h_skip = None

        # Apply EGNN layers
        for i, layer in enumerate(self.layers):
            if self.config.use_checkpoint and self.training:
                h, x, e = checkpoint(layer, h, x, e, use_reentrant=False)
            else:
                h, x, e = layer(h, x, e)

            # Skip connections
            if (i + 1) % 2 == 0:
                if h_skip is not None and skip_idx < len(self.skip_projs):
                    h = self.skip_projs[skip_idx](torch.cat([h, h_skip], dim=-1))
                    skip_idx += 1
                h_skip = h

            if return_intermediates:
                intermediates.append(h)

        # Global context through self-attention
        h_global, _ = self.global_attention(h, h, h)
        h = h + h_global

        h = self.output_norm(h)

        if return_intermediates:
            return h, intermediates
        return h


# ============================================================================
# PARTITION HEAD
# ============================================================================

class CapacityAwareClusterHead(nn.Module):
    """
    Capacity-aware cluster assignment head.

    Produces INVARIANT cluster assignments that:
    1. Respect vehicle capacity constraints
    2. Are balanced across clusters
    3. Maintain spatial compactness
    """

    def __init__(self, config: PartitionConfig):
        super().__init__()
        self.config = config

        # Main cluster projection
        self.cluster_proj = nn.Sequential(
            nn.LayerNorm(config.node_dim),
            nn.Linear(config.node_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(config.hidden_dim // 2, config.n_clusters)
        )

        # Capacity-aware refinement
        self.capacity_refine = nn.Sequential(
            nn.Linear(config.n_clusters + 1, config.hidden_dim // 2),  # +1 for demand
            nn.SiLU(),
            nn.Linear(config.hidden_dim // 2, config.n_clusters)
        )

        # Pairwise affinity for spatial coherence
        self.affinity_net = nn.Sequential(
            nn.Linear(config.node_dim * 2 + 1, config.hidden_dim // 2),  # +1 for distance
            nn.SiLU(),
            nn.Linear(config.hidden_dim // 2, 1)
        )

        # Initialize with small weights
        nn.init.xavier_uniform_(self.cluster_proj[-1].weight, gain=0.1)
        nn.init.zeros_(self.cluster_proj[-1].bias)

    def forward(
        self,
        h: torch.Tensor,
        coords: torch.Tensor,
        demands_norm: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute cluster assignments.

        Args:
            h: Node embeddings (batch, n_nodes, node_dim)
            coords: Coordinates (batch, n_nodes, 2)
            demands_norm: Normalized demands (batch, n_nodes, 1)

        Returns:
            cluster_logits: (batch, n_nodes, n_clusters)
            affinity: Pairwise affinity matrix (batch, n_nodes, n_nodes)
        """
        batch_size, n_nodes, _ = h.shape

        # Main cluster logits
        cluster_logits = self.cluster_proj(h)  # (batch, n_nodes, n_clusters)

        # Capacity-aware refinement
        refine_input = torch.cat([cluster_logits, demands_norm], dim=-1)
        cluster_logits = cluster_logits + 0.1 * self.capacity_refine(refine_input)

        # Depot masking (depot doesn't belong to any cluster)
        # Use -1e4 instead of -1e9 to avoid FP16 overflow in mixed precision
        depot_mask = torch.zeros_like(cluster_logits)
        depot_mask[:, 0, :] = -1e4
        cluster_logits = cluster_logits + depot_mask

        # Compute pairwise affinity for spatial coherence loss
        dist_matrix = torch.cdist(coords, coords).unsqueeze(-1)  # (batch, n, n, 1)
        h_i = h.unsqueeze(2).expand(-1, -1, n_nodes, -1)
        h_j = h.unsqueeze(1).expand(-1, n_nodes, -1, -1)
        affinity_input = torch.cat([h_i, h_j, dist_matrix], dim=-1)
        affinity = self.affinity_net(affinity_input).squeeze(-1)  # (batch, n, n)
        affinity = torch.sigmoid(affinity)

        return cluster_logits, affinity


# ============================================================================
# COMPLETE PARTITION NETWORK
# ============================================================================

class EquivariantPartitionNet(nn.Module):
    """
    Complete E(2)-Equivariant Partition Network for CVRP.

    Architecture:
        Input: coords, demands, capacity
            ↓
        MultiScaleEGNN (geometric encoding)
            ↓
        CapacityAwareClusterHead (cluster assignment)
            ↓
        Output: cluster_logits, affinity
    """

    def __init__(self, config: PartitionConfig):
        super().__init__()
        self.config = config

        # Encoder
        self.encoder = MultiScaleEGNN(config)

        # Cluster head
        self.cluster_head = CapacityAwareClusterHead(config)

    def forward(
        self,
        coords: torch.Tensor,
        demands: torch.Tensor,
        capacity: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            coords: (batch, n_nodes, 2)
            demands: (batch, n_nodes)
            capacity: (batch,) or (batch, 1)

        Returns:
            Dictionary with:
                - cluster_logits: (batch, n_nodes, n_clusters)
                - affinity: (batch, n_nodes, n_nodes)
                - embeddings: (batch, n_nodes, node_dim)
        """
        batch_size, n_nodes, _ = coords.shape

        # Ensure capacity shape
        if capacity.dim() == 1:
            capacity = capacity.unsqueeze(-1)

        # Create invariant features
        is_depot = torch.zeros(batch_size, n_nodes, 1, device=coords.device)
        is_depot[:, 0, :] = 1.0

        demands_norm = demands.unsqueeze(-1) / (capacity.unsqueeze(1) + 1e-8)

        invariant_features = torch.cat([demands_norm, is_depot], dim=-1)

        # Encode
        h = self.encoder(coords, invariant_features)

        # Get cluster assignments
        cluster_logits, affinity = self.cluster_head(h, coords, demands_norm)

        return {
            'cluster_logits': cluster_logits,
            'affinity': affinity,
            'embeddings': h
        }

    def get_clusters(
        self,
        cluster_logits: torch.Tensor,
        demands: torch.Tensor,
        capacity: float,
        strategy: str = 'greedy'
    ) -> List[List[List[int]]]:
        """
        Convert soft assignments to hard clusters.

        Args:
            cluster_logits: (batch, n_nodes, n_clusters)
            demands: (batch, n_nodes)
            capacity: Vehicle capacity
            strategy: 'greedy', 'hungarian', or 'threshold'

        Returns:
            List of cluster assignments per batch
        """
        batch_size = cluster_logits.shape[0]
        probs = F.softmax(cluster_logits, dim=-1)

        all_clusters = []
        for b in range(batch_size):
            if strategy == 'greedy':
                clusters = self._greedy_assignment(probs[b], demands[b], capacity)
            elif strategy == 'hungarian':
                clusters = self._hungarian_assignment(probs[b], demands[b], capacity)
            else:
                clusters = self._threshold_assignment(probs[b], demands[b], capacity)
            all_clusters.append(clusters)

        return all_clusters

    def _greedy_assignment(
        self,
        probs: torch.Tensor,
        demands: torch.Tensor,
        capacity: float
    ) -> List[List[int]]:
        """Greedy capacity-aware assignment."""
        n_nodes, n_clusters = probs.shape
        probs_np = probs.detach().cpu().numpy()
        demands_np = demands.detach().cpu().numpy()

        clusters = [[] for _ in range(n_clusters)]
        cluster_loads = [0.0] * n_clusters

        # Customer indices (skip depot)
        customers = list(range(1, n_nodes))

        # Sort by maximum probability (most confident first)
        max_probs = probs_np[1:].max(axis=1)
        sorted_customers = sorted(customers, key=lambda i: max_probs[i-1], reverse=True)

        for customer in sorted_customers:
            demand = demands_np[customer]
            cluster_prefs = probs_np[customer].argsort()[::-1]

            assigned = False
            for cluster_idx in cluster_prefs:
                if cluster_loads[cluster_idx] + demand <= capacity:
                    clusters[cluster_idx].append(customer)
                    cluster_loads[cluster_idx] += demand
                    assigned = True
                    break

            if not assigned:
                # Find cluster with most remaining capacity
                remaining = [capacity - load for load in cluster_loads]
                best = int(np.argmax(remaining))
                clusters[best].append(customer)
                cluster_loads[best] += demand

        return [c for c in clusters if len(c) > 0]

    def _hungarian_assignment(
        self,
        probs: torch.Tensor,
        demands: torch.Tensor,
        capacity: float
    ) -> List[List[int]]:
        """Hungarian algorithm for optimal assignment."""
        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError:
            return self._greedy_assignment(probs, demands, capacity)

        n_nodes, n_clusters = probs.shape
        probs_np = probs.detach().cpu().numpy()
        demands_np = demands.detach().cpu().numpy()

        # Cost matrix (negative log probability)
        cost_matrix = -np.log(probs_np[1:] + 1e-8)

        # Solve assignment (this gives one-to-one, we need many-to-one)
        # Use as initialization for greedy refinement
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Initialize clusters
        clusters = [[] for _ in range(n_clusters)]
        cluster_loads = [0.0] * n_clusters
        assigned = set()

        # First pass: assign based on Hungarian
        for customer_idx, cluster_idx in zip(row_ind, col_ind):
            customer = customer_idx + 1  # Adjust for depot
            if cluster_loads[cluster_idx] + demands_np[customer] <= capacity:
                clusters[cluster_idx].append(customer)
                cluster_loads[cluster_idx] += demands_np[customer]
                assigned.add(customer)

        # Second pass: greedy for remaining
        remaining = [c for c in range(1, n_nodes) if c not in assigned]
        for customer in remaining:
            demand = demands_np[customer]
            cluster_prefs = probs_np[customer].argsort()[::-1]

            for cluster_idx in cluster_prefs:
                if cluster_loads[cluster_idx] + demand <= capacity:
                    clusters[cluster_idx].append(customer)
                    cluster_loads[cluster_idx] += demand
                    break

        return [c for c in clusters if len(c) > 0]

    def _threshold_assignment(
        self,
        probs: torch.Tensor,
        demands: torch.Tensor,
        capacity: float,
        threshold: float = 0.3
    ) -> List[List[int]]:
        """Threshold-based soft assignment with capacity splitting."""
        n_nodes, n_clusters = probs.shape
        probs_np = probs.detach().cpu().numpy()
        demands_np = demands.detach().cpu().numpy()

        # Hard assignment by argmax
        assignments = probs_np[1:].argmax(axis=1)

        # Group by cluster
        clusters = [[] for _ in range(n_clusters)]
        for customer_idx, cluster_idx in enumerate(assignments):
            clusters[cluster_idx].append(customer_idx + 1)

        # Split clusters that exceed capacity
        final_clusters = []
        for cluster in clusters:
            if not cluster:
                continue

            cluster_demand = sum(demands_np[c] for c in cluster)

            if cluster_demand <= capacity:
                final_clusters.append(cluster)
            else:
                # Split by nearest neighbor ordering
                current = []
                current_load = 0

                for customer in cluster:
                    if current_load + demands_np[customer] <= capacity:
                        current.append(customer)
                        current_load += demands_np[customer]
                    else:
                        if current:
                            final_clusters.append(current)
                        current = [customer]
                        current_load = demands_np[customer]

                if current:
                    final_clusters.append(current)

        return final_clusters


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class PartitionLoss(nn.Module):
    """
    Comprehensive loss function for partition training.

    Components:
    1. Balance loss: Clusters should have similar total demand
    2. Compactness loss: Customers in same cluster should be close
    3. Entropy loss: Encourage confident assignments
    4. Coverage loss: All customers should be assigned
    5. Supervised loss: Match ground-truth routes (if available)
    """

    def __init__(self, config: PartitionConfig):
        super().__init__()
        self.config = config

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        coords: torch.Tensor,
        demands: torch.Tensor,
        capacity: torch.Tensor,
        gt_clusters: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute partition loss.

        Args:
            outputs: Network outputs (cluster_logits, affinity, embeddings)
            coords: (batch, n_nodes, 2)
            demands: (batch, n_nodes)
            capacity: (batch,)
            gt_clusters: Optional ground truth (batch, n_nodes)

        Returns:
            total_loss, loss_dict
        """
        cluster_logits = outputs['cluster_logits']
        affinity = outputs['affinity']

        batch_size, n_nodes, n_clusters = cluster_logits.shape

        # Soft assignments (exclude depot)
        soft_assign = F.softmax(cluster_logits[:, 1:, :], dim=-1)
        customer_coords = coords[:, 1:, :]
        customer_demands = demands[:, 1:]

        if capacity.dim() == 1:
            capacity = capacity.unsqueeze(-1)

        loss_dict = {}
        total_loss = torch.tensor(0.0, device=cluster_logits.device)

        # 1. Balance Loss
        cluster_demands = torch.einsum('bnk,bn->bk', soft_assign, customer_demands)
        target_demand = customer_demands.sum(dim=-1, keepdim=True) / n_clusters
        balance_loss = ((cluster_demands - target_demand) ** 2).mean()

        loss_dict['balance'] = balance_loss.item()
        total_loss = total_loss + self.config.balance_weight * balance_loss

        # 2. Compactness Loss
        # Compute cluster centroids
        cluster_weights = soft_assign.sum(dim=1, keepdim=True) + 1e-8
        weighted_coords = torch.einsum('bnk,bnd->bkd', soft_assign, customer_coords)
        centroids = weighted_coords / cluster_weights.transpose(1, 2)

        # Distance to centroids
        customer_coords_exp = customer_coords.unsqueeze(2)
        centroids_exp = centroids.unsqueeze(1)
        dist_to_centroids = torch.norm(customer_coords_exp - centroids_exp, dim=-1)

        compactness_loss = (soft_assign * dist_to_centroids).sum() / (batch_size * (n_nodes - 1))

        loss_dict['compactness'] = compactness_loss.item()
        total_loss = total_loss + self.config.compactness_weight * compactness_loss

        # 3. Entropy Loss (encourage confident assignments)
        entropy = -(soft_assign * torch.log(soft_assign + 1e-8)).sum(dim=-1).mean()

        loss_dict['entropy'] = entropy.item()
        total_loss = total_loss + self.config.entropy_weight * entropy

        # 4. Coverage Loss (all customers should have high assignment probability)
        max_probs = soft_assign.max(dim=-1).values
        coverage_loss = (1 - max_probs).mean()

        loss_dict['coverage'] = coverage_loss.item()
        total_loss = total_loss + self.config.coverage_weight * coverage_loss

        # 5. Supervised Loss (if ground truth available)
        if gt_clusters is not None:
            gt_customers = gt_clusters[:, 1:]  # Exclude depot
            n_classes = cluster_logits.size(-1)  # Number of clusters
            # Valid mask: assigned (>=0) AND within valid cluster range (<n_classes)
            valid_mask = (gt_customers >= 0) & (gt_customers < n_classes)

            if valid_mask.any():
                supervised_loss = F.cross_entropy(
                    cluster_logits[:, 1:, :][valid_mask],
                    gt_customers[valid_mask].long(),
                    reduction='mean'
                )
                loss_dict['supervised'] = supervised_loss.item()
                total_loss = total_loss + self.config.supervised_weight * supervised_loss

        loss_dict['total'] = total_loss.item()

        return total_loss, loss_dict


class ReinforceLoss(nn.Module):
    """
    REINFORCE-based loss for partition training.

    Directly optimizes for routing distance using policy gradient:
        Loss = (distance - baseline) * (-log_prob)

    This is similar to GLOP's training approach.
    """

    def __init__(self, config: PartitionConfig):
        super().__init__()
        self.config = config

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        coords: torch.Tensor,
        demands: torch.Tensor,
        capacity: torch.Tensor,
        n_samples: int = 1,
        greedy: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute REINFORCE loss.

        Args:
            outputs: Network outputs with 'cluster_logits'
            coords: (batch, n_nodes, 2)
            demands: (batch, n_nodes)
            capacity: (batch,) or scalar
            n_samples: Number of samples for variance reduction
            greedy: If True, use greedy assignment (for evaluation)

        Returns:
            loss, loss_dict
        """
        cluster_logits = outputs['cluster_logits']  # (batch, n_nodes, n_clusters)
        batch_size, n_nodes, n_clusters = cluster_logits.shape

        # Get logits for customers only (exclude depot)
        customer_logits = cluster_logits[:, 1:, :]  # (batch, n_customers, n_clusters)
        n_customers = n_nodes - 1

        # Compute log probabilities
        log_probs = F.log_softmax(customer_logits, dim=-1)
        probs = F.softmax(customer_logits, dim=-1)

        if capacity.dim() == 0:
            capacity = capacity.unsqueeze(0).expand(batch_size)
        elif capacity.dim() == 1 and capacity.size(0) == 1:
            capacity = capacity.expand(batch_size)

        loss_dict = {}

        if greedy:
            # Greedy assignment for evaluation
            assignments = customer_logits.argmax(dim=-1)  # (batch, n_customers)
            distances = self._compute_routing_distances(
                coords, demands, capacity, assignments
            )
            loss_dict['distance'] = distances.mean().item()
            loss_dict['total'] = 0.0
            return torch.tensor(0.0, device=cluster_logits.device), loss_dict

        # Sample multiple times for variance reduction
        all_distances = []
        all_log_probs = []

        for _ in range(n_samples):
            # Sample cluster assignments
            dist = torch.distributions.Categorical(probs=probs)
            assignments = dist.sample()  # (batch, n_customers)

            # Get log probability of sampled assignments
            sample_log_probs = dist.log_prob(assignments)  # (batch, n_customers)
            total_log_prob = sample_log_probs.sum(dim=-1)  # (batch,)

            # Compute routing distance for sampled partition
            with torch.no_grad():
                distances = self._compute_routing_distances(
                    coords, demands, capacity, assignments
                )

            all_distances.append(distances)
            all_log_probs.append(total_log_prob)

        # Stack samples
        distances = torch.stack(all_distances, dim=1)  # (batch, n_samples)
        log_probs_stacked = torch.stack(all_log_probs, dim=1)  # (batch, n_samples)

        # Baseline: per-instance mean (self-critical, following GLOP)
        # GLOP computes baseline as mean over samples for EACH instance separately
        # This is crucial - we compare solutions for the SAME problem, not across problems
        if n_samples > 1:
            # Per-instance baseline: mean over samples for each instance
            baseline = distances.mean(dim=1, keepdim=True).detach()  # (batch, 1)
        else:
            # With only 1 sample, fall back to batch mean (less effective)
            baseline = distances.mean().detach()

        # REINFORCE loss: (distance - baseline) * log_prob
        # Following GLOP: torch.sum((objs-baseline) * log_probs.sum(dim=1)) / bs
        advantage = distances - baseline
        reinforce_loss = (advantage * log_probs_stacked).sum() / (batch_size * n_samples)

        loss_dict['distance'] = distances.mean().item()
        loss_dict['baseline'] = baseline.mean().item() if baseline.numel() > 1 else baseline.item()
        loss_dict['advantage'] = advantage.mean().item()
        loss_dict['total'] = reinforce_loss.item()

        return reinforce_loss, loss_dict

    def _compute_routing_distances(
        self,
        coords: torch.Tensor,
        demands: torch.Tensor,
        capacity: torch.Tensor,
        assignments: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute total routing distance for given cluster assignments.

        Args:
            coords: (batch, n_nodes, 2) - includes depot at index 0
            demands: (batch, n_nodes)
            capacity: (batch,)
            assignments: (batch, n_customers) - cluster assignment per customer

        Returns:
            distances: (batch,) total routing distance
        """
        batch_size = coords.size(0)
        n_clusters = self.config.n_clusters
        device = coords.device

        dist_list = []

        for b in range(batch_size):
            depot = coords[b, 0].cpu().numpy()
            customer_coords = coords[b, 1:].cpu().numpy()
            customer_demands = demands[b, 1:].cpu().numpy()
            cap = capacity[b].item()
            cluster_assigns = assignments[b].cpu().numpy()

            total_dist = 0.0

            # Process each cluster
            for k in range(n_clusters):
                cluster_mask = (cluster_assigns == k)
                if not cluster_mask.any():
                    continue

                cluster_indices = np.where(cluster_mask)[0]
                cluster_coords = customer_coords[cluster_indices]
                cluster_demands = customer_demands[cluster_indices]

                # Split cluster into capacity-feasible routes
                routes = self._split_by_capacity(
                    cluster_indices, cluster_demands, cap
                )

                # Route each sub-route
                for route_indices in routes:
                    if len(route_indices) == 0:
                        continue
                    route_coords = customer_coords[route_indices]
                    route_dist = self._route_nearest_neighbor(depot, route_coords)
                    total_dist += route_dist

            dist_list.append(total_dist)

        return torch.tensor(dist_list, device=device, dtype=torch.float32)

    def _split_by_capacity(
        self,
        indices: np.ndarray,
        demands: np.ndarray,
        capacity: float
    ) -> List[List[int]]:
        """Split customers into capacity-feasible groups."""
        routes = []
        current_route = []
        current_load = 0.0

        for idx, demand in zip(indices, demands):
            if current_load + demand > capacity and current_route:
                routes.append(current_route)
                current_route = [idx]
                current_load = demand
            else:
                current_route.append(idx)
                current_load += demand

        if current_route:
            routes.append(current_route)

        return routes

    def _route_nearest_neighbor(
        self,
        depot: np.ndarray,
        customer_coords: np.ndarray
    ) -> float:
        """Compute route distance using nearest neighbor heuristic."""
        if len(customer_coords) == 0:
            return 0.0

        n = len(customer_coords)
        visited = [False] * n
        route = []

        # Start from depot, find nearest customer
        current = depot
        total_dist = 0.0

        for _ in range(n):
            best_dist = float('inf')
            best_idx = -1

            for i in range(n):
                if not visited[i]:
                    dist = np.linalg.norm(current - customer_coords[i])
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = i

            if best_idx >= 0:
                visited[best_idx] = True
                total_dist += best_dist
                current = customer_coords[best_idx]
                route.append(best_idx)

        # Return to depot
        total_dist += np.linalg.norm(current - depot)

        return total_dist


# ============================================================================
# EQUIVARIANCE VERIFICATION
# ============================================================================

def verify_partition_equivariance(
    model: EquivariantPartitionNet,
    batch_size: int = 2,
    n_nodes: int = 51,
    n_tests: int = 5,
    device: str = 'cuda'
) -> Dict[str, bool]:
    """
    Verify E(2) equivariance of the partition network.

    Tests that cluster assignments are invariant to:
    1. Rotation
    2. Translation
    3. Reflection
    """
    model.eval()
    results = {'rotation': True, 'translation': True, 'reflection': True}
    tolerance = 1e-3

    with torch.no_grad():
        # Generate test data
        coords = torch.rand(batch_size, n_nodes, 2, device=device)
        coords[:, 0, :] = 0.5  # Depot at center

        demands = torch.randint(1, 10, (batch_size, n_nodes), device=device).float()
        demands[:, 0] = 0

        capacity = torch.full((batch_size,), 40.0, device=device)

        # Original forward pass
        original_out = model(coords, demands, capacity)
        original_probs = F.softmax(original_out['cluster_logits'], dim=-1)

        for _ in range(n_tests):
            # Test 1: Rotation
            theta = torch.rand(1).item() * 2 * np.pi
            R = torch.tensor([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]
            ], dtype=torch.float32, device=device)

            coords_rot = coords @ R.T
            rot_out = model(coords_rot, demands, capacity)
            rot_probs = F.softmax(rot_out['cluster_logits'], dim=-1)

            if (original_probs - rot_probs).abs().max() > tolerance:
                results['rotation'] = False

            # Test 2: Translation
            t = torch.randn(1, 1, 2, device=device) * 5
            coords_trans = coords + t

            trans_out = model(coords_trans, demands, capacity)
            trans_probs = F.softmax(trans_out['cluster_logits'], dim=-1)

            if (original_probs - trans_probs).abs().max() > tolerance:
                results['translation'] = False

            # Test 3: Reflection
            reflect = torch.tensor([[-1, 0], [0, 1]], dtype=torch.float32, device=device)
            coords_ref = coords @ reflect.T

            ref_out = model(coords_ref, demands, capacity)
            ref_probs = F.softmax(ref_out['cluster_logits'], dim=-1)

            if (original_probs - ref_probs).abs().max() > tolerance:
                results['reflection'] = False

    return results


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def compute_route_distance(coords: np.ndarray, routes: List[List[int]]) -> float:
    """Compute total CVRP solution distance."""
    total = 0.0
    for route in routes:
        if not route:
            continue
        total += np.linalg.norm(coords[0] - coords[route[0]])
        for i in range(len(route) - 1):
            total += np.linalg.norm(coords[route[i]] - coords[route[i+1]])
        total += np.linalg.norm(coords[route[-1]] - coords[0])
    return total


def nearest_neighbor_route(coords: np.ndarray, customers: List[int]) -> List[int]:
    """Simple nearest neighbor routing for a cluster."""
    if len(customers) <= 1:
        return customers

    route = []
    remaining = set(customers)
    current_pos = coords[0]  # Start at depot

    while remaining:
        nearest = min(remaining, key=lambda c: np.linalg.norm(current_pos - coords[c]))
        route.append(nearest)
        current_pos = coords[nearest]
        remaining.remove(nearest)

    return route


# ============================================================================
# HEATMAP-BASED EGNN FOR GLOP-STYLE TRAINING
# ============================================================================

@dataclass
class HeatmapConfig:
    """Configuration for heatmap-based partition network."""
    # Architecture
    n_layers: int = 8
    node_dim: int = 64
    edge_dim: int = 64
    hidden_dim: int = 128
    n_heads: int = 4
    dropout: float = 0.0

    # Graph construction
    k_sparse: int = 50  # Number of neighbors per node

    # Training
    update_coords: bool = True


class HeatmapEGNNLayer(nn.Module):
    """
    Simplified EGNN layer for heatmap prediction.

    Optimized for producing edge-level outputs (heatmap).
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        update_coords: bool = True,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.update_coords = update_coords

        # Message MLP: [h_i, h_j, d_ij, e_ij] -> message
        self.msg_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + 1 + edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # Node update
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, node_dim),
        )

        # Edge update
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, edge_dim),
        )

        # Coordinate update (scalar weights)
        if update_coords:
            self.coord_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.SiLU(),
                nn.Linear(hidden_dim // 2, 1, bias=False),
            )
            nn.init.xavier_uniform_(self.coord_mlp[-1].weight, gain=0.01)

        self.node_norm = nn.LayerNorm(node_dim)
        self.edge_norm = nn.LayerNorm(edge_dim)

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            h: Node features (n_nodes, node_dim)
            x: Coordinates (n_nodes, 2)
            edge_index: (2, n_edges)
            edge_attr: Edge features (n_edges, edge_dim)

        Returns:
            h_new, x_new, edge_attr_new
        """
        src, dst = edge_index
        n_nodes = h.size(0)

        # Compute distances (INVARIANT)
        rel_vec = x[dst] - x[src]
        dist = rel_vec.norm(dim=-1, keepdim=True)

        # Build message input (all INVARIANT)
        msg_input = torch.cat([h[src], h[dst], dist, edge_attr], dim=-1)
        msg = self.msg_mlp(msg_input)

        # Aggregate messages
        agg_msg = torch.zeros(n_nodes, self.hidden_dim, device=h.device, dtype=h.dtype)
        agg_msg.index_add_(0, dst, msg)

        # Update nodes
        h_new = h + self.node_mlp(torch.cat([h, agg_msg], dim=-1))
        h_new = self.node_norm(h_new)

        # Update edges
        edge_attr_new = edge_attr + self.edge_mlp(torch.cat([edge_attr, msg], dim=-1))
        edge_attr_new = self.edge_norm(edge_attr_new)

        # Update coordinates (EQUIVARIANT)
        if self.update_coords:
            coord_weights = self.coord_mlp(msg)
            coord_weights = torch.tanh(coord_weights)

            rel_dir = rel_vec / (dist + 1e-8)
            weighted_vec = coord_weights * rel_dir

            coord_delta = torch.zeros_like(x)
            coord_delta.index_add_(0, dst, weighted_vec)

            # Normalize by edge count
            edge_count = torch.zeros(n_nodes, 1, device=x.device, dtype=x.dtype)
            edge_count.index_add_(0, dst, torch.ones(len(src), 1, device=x.device, dtype=x.dtype))
            edge_count = edge_count.clamp(min=1)

            x_new = x + coord_delta / edge_count
        else:
            x_new = x

        return h_new, x_new, edge_attr_new


class HeatmapEGNN(nn.Module):
    """
    E(2)-Equivariant network that outputs a heatmap for sequential sampling.

    This network follows GLOP's approach:
    1. Process graph with EGNN layers
    2. Output edge-level probabilities (heatmap)
    3. Used with sequential sampling for REINFORCE training

    The heatmap H[i,j] represents the probability of transitioning from node i to j.
    """

    def __init__(self, config: HeatmapConfig):
        super().__init__()
        self.config = config

        # Input embeddings
        self.node_embed = nn.Sequential(
            nn.Linear(2, config.node_dim),  # (demand/cap, dist_to_depot)
            nn.SiLU(),
            nn.Linear(config.node_dim, config.node_dim),
        )

        self.edge_embed = nn.Sequential(
            nn.Linear(2, config.edge_dim),  # (distance, affinity)
            nn.SiLU(),
            nn.Linear(config.edge_dim, config.edge_dim),
        )

        # EGNN layers
        self.layers = nn.ModuleList()
        for _ in range(config.n_layers):
            self.layers.append(
                HeatmapEGNNLayer(
                    node_dim=config.node_dim,
                    edge_dim=config.edge_dim,
                    hidden_dim=config.hidden_dim,
                    update_coords=config.update_coords,
                )
            )

        # Skip connection projections
        self.skip_projs = nn.ModuleList()
        for _ in range(config.n_layers // 2):
            self.skip_projs.append(nn.Linear(config.node_dim * 2, config.node_dim))

        # Heatmap prediction head
        self.heatmap_head = nn.Sequential(
            nn.Linear(config.edge_dim, config.edge_dim),
            nn.SiLU(),
            nn.Linear(config.edge_dim, config.edge_dim // 2),
            nn.SiLU(),
            nn.Linear(config.edge_dim // 2, 1),
        )

        # Initialize output to small values
        nn.init.xavier_uniform_(self.heatmap_head[-1].weight, gain=0.1)
        nn.init.zeros_(self.heatmap_head[-1].bias)

    def forward(
        self,
        coords: torch.Tensor,
        demands: torch.Tensor,
        capacity: float,
        edge_index: torch.Tensor,
        return_embeddings: bool = False
    ) -> torch.Tensor:
        """
        Forward pass producing heatmap.

        Args:
            coords: (n_nodes, 2) node coordinates
            demands: (n_nodes,) demand per node
            capacity: Vehicle capacity
            edge_index: (2, n_edges) sparse edge indices
            return_embeddings: Whether to return node embeddings

        Returns:
            heatmap: (n_nodes, n_nodes) transition probability matrix
        """
        n_nodes = coords.size(0)
        device = coords.device
        k_sparse = self.config.k_sparse

        # Compute node features (INVARIANT)
        depot = coords[0:1]
        dist_to_depot = torch.norm(coords - depot, dim=1, keepdim=True)
        demand_norm = (demands / capacity).unsqueeze(1)
        node_feats = torch.cat([demand_norm, dist_to_depot], dim=1)

        # Compute edge features (INVARIANT)
        src, dst = edge_index
        edge_dist = torch.norm(coords[dst] - coords[src], dim=1, keepdim=True)
        max_dist = edge_dist.max() + 1e-8
        edge_affinity = 1 - edge_dist / max_dist
        edge_feats = torch.cat([edge_dist, edge_affinity], dim=1)

        # Embed
        h = self.node_embed(node_feats)
        e = self.edge_embed(edge_feats)
        x = coords.clone()

        # EGNN layers with skip connections
        skip_h = None
        skip_e = None

        for i, layer in enumerate(self.layers):
            if i % 2 == 0:
                skip_h = h
                skip_e = e

            h, x, e = layer(h, x, edge_index, e)

            if i % 2 == 1 and i < self.config.n_layers - 1:
                proj_idx = i // 2
                if proj_idx < len(self.skip_projs):
                    h = self.skip_projs[proj_idx](torch.cat([h, skip_h], dim=-1))
                e = e + skip_e

        # Predict heatmap logits
        logits = self.heatmap_head(e).squeeze(-1)  # (n_edges,)

        # Reshape to per-source softmax
        logits_matrix = torch.full((n_nodes, n_nodes), float('-inf'), device=device)
        logits_matrix[src, dst] = logits

        # Softmax per row (per source node)
        heatmap = F.softmax(logits_matrix, dim=1)

        if return_embeddings:
            return heatmap, h
        return heatmap

    @staticmethod
    def build_knn_graph(coords: torch.Tensor, k: int) -> torch.Tensor:
        """
        Build k-NN graph based on distances.

        Args:
            coords: (n_nodes, 2)
            k: Number of neighbors

        Returns:
            edge_index: (2, n_nodes * k)
        """
        n_nodes = coords.size(0)
        device = coords.device

        # Compute distance matrix
        dist_matrix = torch.cdist(coords, coords)

        # Get k nearest neighbors (excluding self)
        dist_matrix.fill_diagonal_(float('inf'))
        _, indices = dist_matrix.topk(k, dim=1, largest=False)

        # Build edge index
        src = torch.arange(n_nodes, device=device).unsqueeze(1).expand(-1, k).flatten()
        dst = indices.flatten()

        edge_index = torch.stack([src, dst], dim=0)
        return edge_index


def create_heatmap_model(
    k_sparse: int = 50,
    n_layers: int = 8,
    node_dim: int = 64,
    **kwargs
) -> HeatmapEGNN:
    """Create heatmap-based EGNN model."""
    config = HeatmapConfig(
        k_sparse=k_sparse,
        n_layers=n_layers,
        node_dim=node_dim,
        **kwargs
    )
    return HeatmapEGNN(config)


# ============================================================================
# MODEL CREATION
# ============================================================================

def create_partition_model(
    n_clusters: int = 10,
    n_layers: int = 8,
    hidden_dim: int = 256,
    **kwargs
) -> EquivariantPartitionNet:
    """Create partition model with specified configuration."""
    config = PartitionConfig(
        n_clusters=n_clusters,
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        **kwargs
    )
    return EquivariantPartitionNet(config)


if __name__ == "__main__":
    # Quick test
    print("Testing EDISCO-Partition Model...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    config = PartitionConfig(n_clusters=10, n_layers=6, hidden_dim=128)
    model = EquivariantPartitionNet(config).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # Test forward pass
    batch_size = 4
    n_nodes = 201  # 200 customers + depot

    coords = torch.rand(batch_size, n_nodes, 2, device=device)
    demands = torch.randint(1, 10, (batch_size, n_nodes), device=device).float()
    demands[:, 0] = 0
    capacity = torch.full((batch_size,), 80.0, device=device)

    outputs = model(coords, demands, capacity)

    print(f"Cluster logits shape: {outputs['cluster_logits'].shape}")
    print(f"Affinity shape: {outputs['affinity'].shape}")
    print(f"Embeddings shape: {outputs['embeddings'].shape}")

    # Test equivariance
    print("\nVerifying equivariance...")
    results = verify_partition_equivariance(model, device=device)
    for test, passed in results.items():
        print(f"  {test}: {'PASSED' if passed else 'FAILED'}")

    print("\nTest complete!")
