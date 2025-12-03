"""
E(2)-Equivariant Graph Neural Network for CVRP Partition.

This module implements EGNN layers that respect E(2) symmetry (rotations and translations).
The key insight is that messages are computed using only invariant quantities (distances),
while coordinate updates are equivariant (scalar * direction).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EGNNLayer(nn.Module):
    """
    E(2)-Equivariant GNN Layer.

    Processes graph with:
    - Node features h (invariant)
    - Node coordinates x (equivariant)
    - Edge features e (invariant)

    Key properties:
    - Messages computed from invariant quantities only (distances, not angles)
    - Coordinate updates: x_new = x + alpha * sum_j(scalar_ij * (x_j - x_i) / ||x_j - x_i||)
    """

    def __init__(self, node_dim, edge_dim, hidden_dim, coord_dim=2,
                 coord_update_alpha=0.1, weight_temp=10.0):
        super().__init__()
        self.coord_dim = coord_dim
        self.hidden_dim = hidden_dim
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.coord_update_alpha = coord_update_alpha
        self.weight_temp = weight_temp

        # Message network: h_i, h_j, e_ij, ||x_i - x_j|| -> m_ij
        self.message_mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Coordinate update network: m_ij -> scalar (no bias to preserve equivariance)
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False)
        )

        # Node update network
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, node_dim)
        )

        # Edge update network
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, edge_dim)
        )

        self.node_norm = nn.LayerNorm(node_dim)
        self.edge_norm = nn.LayerNorm(edge_dim)

    def forward(self, h, x, e, edge_index):
        """
        Forward pass with sparse edge representation.

        Args:
            h: Node features (n_nodes, node_dim)
            x: Node coordinates (n_nodes, coord_dim)
            e: Edge features (n_edges, edge_dim)
            edge_index: Edge indices (2, n_edges)

        Returns:
            h_new, x_new, e_new: Updated features and coordinates
        """
        n_nodes = h.shape[0]
        row, col = edge_index[0], edge_index[1]

        # Compute coordinate differences and distances (invariant)
        x_diff = x[col] - x[row]  # (n_edges, coord_dim)
        distances = torch.norm(x_diff, dim=-1, keepdim=True)  # (n_edges, 1)

        # Prepare node features for edges
        h_i = h[row]  # (n_edges, node_dim)
        h_j = h[col]  # (n_edges, node_dim)

        # Compute messages
        msg_input = torch.cat([h_i, h_j, e, distances], dim=-1)
        messages = self.message_mlp(msg_input)  # (n_edges, hidden_dim)

        # Coordinate updates (equivariant)
        coord_weights = self.coord_mlp(messages)  # (n_edges, 1)
        coord_weights = torch.tanh(coord_weights / self.weight_temp)

        # Direction: (x_j - x_i) / ||x_j - x_i||
        direction = x_diff / (distances + 1e-8)
        x_update = coord_weights * direction  # (n_edges, coord_dim)

        # Aggregate coordinate updates
        x_agg = torch.zeros(n_nodes, self.coord_dim, device=x.device)
        x_agg.index_add_(0, row, x_update)
        x_new = x + self.coord_update_alpha * x_agg

        # Aggregate messages for nodes
        h_agg = torch.zeros(n_nodes, self.hidden_dim, device=h.device)
        h_agg.index_add_(0, row, messages)
        h_new = self.node_norm(h + self.node_mlp(torch.cat([h, h_agg], dim=-1)))

        # Update edge features
        e_new = self.edge_norm(e + self.edge_mlp(torch.cat([e, messages], dim=-1)))

        return h_new, x_new, e_new


class EGNN(nn.Module):
    """
    E(2)-Equivariant Graph Neural Network for CVRP Partition.

    Takes a sparse graph and outputs edge embeddings that can be converted
    to a heatmap for sequential sampling.
    """

    def __init__(self, n_layers=12, hidden_dim=128, node_dim=64, edge_dim=64,
                 node_input_dim=2, edge_input_dim=2, coord_dim=2,
                 coord_update_alpha=0.1, weight_temp=10.0):
        """
        Args:
            n_layers: Number of EGNN layers
            hidden_dim: Hidden dimension in message MLPs
            node_dim: Node embedding dimension
            edge_dim: Edge embedding dimension
            node_input_dim: Input node feature dimension (demand, distance_from_depot)
            edge_input_dim: Input edge feature dimension (distance, affinity)
            coord_dim: Coordinate dimension (2 for 2D)
        """
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.coord_dim = coord_dim

        # Input embeddings
        self.node_embed = nn.Linear(node_input_dim, node_dim)
        self.edge_embed = nn.Linear(edge_input_dim, edge_dim)

        # EGNN layers
        self.layers = nn.ModuleList([
            EGNNLayer(node_dim, edge_dim, hidden_dim, coord_dim,
                     coord_update_alpha, weight_temp)
            for _ in range(n_layers)
        ])

        # Output head for edge probabilities
        self.output_head = nn.Sequential(
            nn.LayerNorm(edge_dim),
            nn.Linear(edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, node_features, edge_features, coords, edge_index):
        """
        Forward pass.

        Args:
            node_features: (n_nodes, node_input_dim) - invariant features (demand, r)
            edge_features: (n_edges, edge_input_dim) - invariant features (distance, affinity)
            coords: (n_nodes, 2) - node coordinates
            edge_index: (2, n_edges) - sparse edge indices

        Returns:
            edge_logits: (n_edges,) - unnormalized log probabilities for edges
        """
        # Embed inputs
        h = self.node_embed(node_features)  # (n_nodes, node_dim)
        e = self.edge_embed(edge_features)  # (n_edges, edge_dim)
        x = coords.clone()

        # Apply EGNN layers
        for layer in self.layers:
            h, x, e = layer(h, x, e, edge_index)

        # Output edge logits
        edge_logits = self.output_head(e).squeeze(-1)  # (n_edges,)

        return edge_logits

    def get_heatmap(self, node_features, edge_features, coords, edge_index, n_nodes, k_sparse):
        """
        Get full heatmap matrix from edge logits.

        Args:
            node_features, edge_features, coords, edge_index: Graph data
            n_nodes: Number of nodes
            k_sparse: Number of neighbors per node

        Returns:
            heatmap: (n_nodes, n_nodes) - transition probability matrix
        """
        edge_logits = self.forward(node_features, edge_features, coords, edge_index)

        # Reshape to (n_nodes, k_sparse) for per-row softmax
        edge_logits_2d = edge_logits.reshape(n_nodes, k_sparse)
        edge_probs = F.softmax(edge_logits_2d, dim=1).flatten()

        # Build sparse heatmap
        heatmap = torch.zeros(n_nodes, n_nodes, device=edge_logits.device)
        heatmap[edge_index[0], edge_index[1]] = edge_probs

        # Normalize rows to sum to 1 (handle rows with no outgoing edges)
        row_sums = heatmap.sum(dim=1, keepdim=True)
        row_sums = torch.where(row_sums > 0, row_sums, torch.ones_like(row_sums))
        heatmap = heatmap / row_sums

        return heatmap
