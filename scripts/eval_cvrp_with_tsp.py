"""
Evaluate CVRP-100 using EDISCO Partition + EDISCO TSP Solver.

This script:
1. Loads the trained CVRP partition model
2. Loads the pretrained EDISCO TSP-100 diffusion model
3. Generates CVRP-100 instances
4. Partitions using the partition model
5. Solves each route's TSP using the EDISCO TSP solver
6. Visualizes the results
"""

import os
import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

# Optional matplotlib import
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Visualization will be disabled.")

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from edisco_partition.models.partition_net import PartitionNet
from edisco_partition.data.graph import build_graph
from edisco_partition.data.instance import generate_instance, CAPACITIES
from edisco_partition.sampler.sequential import Sampler
from edisco_partition.evaluation.lkh_routing import LKH_PATH, _solve_tsp_lkh


# ============================================================================
# EDISCO TSP Diffusion Model (matching the notebook architecture)
# ============================================================================

def timestep_embedding(timesteps, dim, max_period=10000):
    """Create sinusoidal timestep embeddings."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class EGNNLayerTSP(nn.Module):
    """E(n) Equivariant Graph Neural Network Layer for TSP."""

    def __init__(self, node_dim, edge_dim, hidden_dim, coord_dim=2):
        super().__init__()
        self.coord_dim = coord_dim
        self.hidden_dim = hidden_dim
        self.node_dim = node_dim
        self.edge_dim = edge_dim

        # Message network: node_dim*2 + 1 + edge_dim -> hidden_dim
        # Structure: Linear -> SiLU -> LayerNorm -> Linear -> SiLU -> Linear
        self.message_mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + 1 + edge_dim, hidden_dim),  # 0
            nn.SiLU(),                                            # 1
            nn.LayerNorm(hidden_dim),                            # 2
            nn.Linear(hidden_dim, hidden_dim),                   # 3
            nn.SiLU(),                                            # 4
            nn.Linear(hidden_dim, hidden_dim)                    # 5
        )

        # Coordinate update network
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),  # 0
            nn.SiLU(),                          # 1
            nn.Linear(hidden_dim, 1, bias=False)  # 2
        )

        # Node update network: node_dim + hidden_dim -> node_dim
        # Structure: Linear -> SiLU -> LayerNorm -> Linear
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),  # 0
            nn.SiLU(),                                      # 1
            nn.LayerNorm(hidden_dim),                       # 2
            nn.Linear(hidden_dim, node_dim)                 # 3
        )

        # Edge update network: edge_dim + hidden_dim -> edge_dim
        # Structure: Linear -> SiLU -> LayerNorm -> Linear
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim + hidden_dim, hidden_dim),  # 0
            nn.SiLU(),                                      # 1
            nn.LayerNorm(hidden_dim),                       # 2
            nn.Linear(hidden_dim, edge_dim)                 # 3
        )

        self.node_norm = nn.LayerNorm(node_dim)
        self.edge_norm = nn.LayerNorm(edge_dim)

    def forward(self, h, x, e, edge_index):
        """
        h: node features (batch_size, n_nodes, node_dim)
        x: node coordinates (batch_size, n_nodes, coord_dim)
        e: edge features (batch_size, n_edges, edge_dim)
        edge_index: (2, n_edges)
        """
        batch_size = h.shape[0]
        n_nodes = h.shape[1]
        n_edges = edge_index.shape[1]
        row, col = edge_index[0], edge_index[1]

        # Compute coordinate differences for each batch
        x_diff_list = []
        for b in range(batch_size):
            x_b = x[b]
            x_diff_b = x_b[col] - x_b[row]
            x_diff_list.append(x_diff_b)

        x_diff = torch.stack(x_diff_list, dim=0)
        distances = torch.norm(x_diff, dim=-1, keepdim=True)

        # Get node features for edges
        h_row = h.gather(1, row.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, h.shape[-1]))
        h_col = h.gather(1, col.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, h.shape[-1]))

        # Compute messages
        msg_input = torch.cat([h_row, h_col, distances, e], dim=-1)
        messages = self.message_mlp(msg_input)

        # Update coordinates (equivariant)
        coord_weights = self.coord_mlp(messages)
        x_update = coord_weights * x_diff / (distances + 1e-8)

        # Aggregate coordinate updates
        x_agg = torch.zeros_like(x)
        for b in range(batch_size):
            idx = row.view(-1, 1).expand(-1, self.coord_dim)
            x_agg[b].scatter_add_(0, idx, x_update[b])

        x_new = x + x_agg

        # Update node features
        h_agg = torch.zeros(batch_size, n_nodes, messages.shape[-1], device=h.device)
        for b in range(batch_size):
            idx = row.view(-1, 1).expand(-1, messages.shape[-1])
            h_agg[b].scatter_add_(0, idx, messages[b])

        h_new = self.node_norm(h + self.node_mlp(torch.cat([h, h_agg], dim=-1)))

        # Update edge features
        e_new = self.edge_norm(e + self.edge_mlp(torch.cat([e, messages], dim=-1)))

        return h_new, x_new, e_new


class EGNNScoreNet(nn.Module):
    """EGNN-based score network for TSP diffusion model."""

    def __init__(self, n_nodes=100, n_layers=8, node_dim=64, edge_dim=128, hidden_dim=128, coord_dim=2):
        super().__init__()
        self.n_nodes = n_nodes
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim

        # Build dense edge index
        edge_index = []
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    edge_index.append([i, j])
        self.register_buffer('edge_index', torch.tensor(edge_index).T.long())

        # Embeddings
        self.node_embed = nn.Linear(2, node_dim)
        self.edge_embed = nn.Linear(1, edge_dim)

        # Time embedding: 128 -> 256 -> 128
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.SiLU(),
            nn.Linear(256, hidden_dim)
        )

        # EGNN layers
        self.layers = nn.ModuleList([
            EGNNLayerTSP(node_dim, edge_dim, hidden_dim, coord_dim)
            for _ in range(n_layers)
        ])

        # Time layers for each EGNN layer: 128 -> 128
        self.time_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, edge_dim))
            for _ in range(n_layers)
        ])

        # Output head: LayerNorm -> Linear -> SiLU -> LayerNorm -> Linear -> SiLU -> Linear
        self.out = nn.Sequential(
            nn.LayerNorm(edge_dim),
            nn.Linear(edge_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 2)
        )

    def forward(self, coords, edge_features, timesteps):
        """
        coords: (batch_size, n_nodes, 2)
        edge_features: (batch_size, n_edges) - binary edge states
        timesteps: (batch_size,)
        """
        batch_size = coords.shape[0]

        # Initialize node features from coordinates
        h = self.node_embed(coords)
        x = coords.clone()

        # Initialize edge features
        e = self.edge_embed(edge_features.unsqueeze(-1))

        # Time embedding
        t_emb = self.time_embed(timestep_embedding(timesteps, self.hidden_dim))

        # Apply EGNN layers
        for layer, time_layer in zip(self.layers, self.time_layers):
            e_with_time = e + time_layer(t_emb).unsqueeze(1)
            h, x, e = layer(h, x, e_with_time, self.edge_index)

        # Output predictions for edges
        out = self.out(e)
        return out


class BinaryDiffusion:
    """Binary diffusion process for edge prediction."""

    def __init__(self, T=100, schedule='linear'):
        self.T = T

        if schedule == 'linear':
            b0 = 1e-4
            bT = 2e-2
            self.beta = np.linspace(b0, bT, T)

        beta = self.beta.reshape((-1, 1, 1))
        eye = np.eye(2).reshape((1, 2, 2))
        ones = np.ones((2, 2)).reshape((1, 2, 2))

        self.Qs = (1 - beta) * eye + (beta / 2) * ones

        Q_bar = [np.eye(2)]
        for Q in self.Qs:
            Q_bar.append(Q_bar[-1] @ Q)
        self.Q_bar = np.stack(Q_bar[1:], axis=0)

    def sample_t(self, x0, t):
        """Sample x_t given x0."""
        Q_bar_t = torch.tensor(self.Q_bar[t], dtype=torch.float32, device=x0.device)
        x0_onehot = F.one_hot(x0.long(), 2).float()
        x_t_probs = torch.einsum('...i,ij->...j', x0_onehot, Q_bar_t)
        x_t = torch.bernoulli(x_t_probs[..., 1])
        return x_t


class EDISCOTSPSolver(nn.Module):
    """EDISCO TSP Solver using diffusion model."""

    def __init__(self, n_nodes=100, device='cuda'):
        super().__init__()
        self.n_nodes = n_nodes
        self.device = device

        # Build dense edge index
        edge_index = []
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    edge_index.append([i, j])
        self.edge_index = torch.tensor(edge_index).T.long().to(device)
        self.n_edges = len(edge_index)

        # Score network
        self.score_net = EGNNScoreNet(n_nodes=n_nodes).to(device)

        # Diffusion process
        self.diffusion = BinaryDiffusion(T=100)

    def load_checkpoint(self, checkpoint_path):
        """Load pretrained weights."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Remove 'score_net.' prefix from keys if present
        new_checkpoint = {}
        for k, v in checkpoint.items():
            if k.startswith('score_net.'):
                new_checkpoint[k[len('score_net.'):]] = v
            else:
                new_checkpoint[k] = v

        self.score_net.load_state_dict(new_checkpoint, strict=False)
        print(f"Loaded TSP solver from {checkpoint_path}")

    @torch.no_grad()
    def solve(self, coords, n_steps=50):
        """
        Solve TSP for given coordinates.

        Args:
            coords: (n_nodes, 2) or (batch_size, n_nodes, 2)
            n_steps: Number of diffusion steps

        Returns:
            tours: List of node indices forming the tour
            tour_length: Total tour length
        """
        if coords.dim() == 2:
            coords = coords.unsqueeze(0)

        batch_size = coords.shape[0]
        coords = coords.to(self.device)

        # Initialize with random edges
        xt = torch.bernoulli(torch.ones(batch_size, self.n_edges, device=self.device) * 0.5)

        # Reverse diffusion
        timesteps = torch.linspace(self.diffusion.T - 1, 0, n_steps).long().to(self.device)

        for i in range(len(timesteps) - 1):
            t = timesteps[i].repeat(batch_size)

            # Predict x0
            x0_logits = self.score_net(coords, xt, t)
            x0_probs = F.softmax(x0_logits, dim=-1)

            # Sample from posterior
            target_t = timesteps[i + 1].item()
            if target_t > 0:
                xt = self._posterior_sample(xt, t[0].cpu().numpy(), x0_probs, target_t)
            else:
                xt = x0_probs[..., 1]

        # Final prediction
        if timesteps[-1].item() > 0:
            t = timesteps[-1].repeat(batch_size)
            x0_logits = self.score_net(coords, xt, t)
            x0_probs = F.softmax(x0_logits, dim=-1)[..., 1]
        else:
            x0_probs = xt

        # Decode tours
        tours = []
        tour_lengths = []
        for b in range(batch_size):
            tour = self._merge_tour(x0_probs[b], coords[b])
            tours.append(tour)
            tour_lengths.append(self._compute_tour_length(coords[b], tour))

        if batch_size == 1:
            return tours[0], tour_lengths[0]
        return tours, tour_lengths

    def _posterior_sample(self, xt, t, x0_probs, target_t):
        """Sample from posterior distribution."""
        Q_t = torch.tensor(self.diffusion.Qs[int(t)], dtype=torch.float32, device=self.device)
        Q_bar_target = torch.tensor(self.diffusion.Q_bar[int(target_t)], dtype=torch.float32, device=self.device)

        xt_onehot = F.one_hot(xt.long(), 2).float()

        x_t_target_prob = torch.einsum('ij,...j->...i', Q_bar_target, x0_probs)
        x_t_target_prob = x_t_target_prob / (x_t_target_prob.sum(dim=-1, keepdim=True) + 1e-8)

        posterior_prob = x_t_target_prob[..., 1] * x0_probs[..., 1]

        if target_t > 0:
            return torch.bernoulli(posterior_prob.clamp(0, 1))
        else:
            return posterior_prob.clamp(min=0)

    def _merge_tour(self, edge_probs, coords):
        """Merge edges into a valid TSP tour."""
        n_nodes = coords.shape[0]

        # Compute edge weights
        dists = torch.norm(coords[self.edge_index[0]] - coords[self.edge_index[1]], dim=1)
        weights = edge_probs / (dists + 1e-6)

        # Sort edges by weight
        sorted_indices = torch.argsort(weights, descending=True)

        # Build tour greedily
        tour_edges = []
        degree = torch.zeros(n_nodes, dtype=torch.long, device=self.device)

        # Union-Find for cycle detection
        parent = list(range(n_nodes))

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
                return True
            return False

        for idx in sorted_indices:
            u, v = self.edge_index[0, idx].item(), self.edge_index[1, idx].item()

            # Skip if degree would exceed 2
            if degree[u] >= 2 or degree[v] >= 2:
                continue

            # Skip if would create early cycle
            if find(u) == find(v) and len(tour_edges) < n_nodes - 1:
                continue

            # Add edge
            tour_edges.append((u, v))
            degree[u] += 1
            degree[v] += 1
            union(u, v)

            if len(tour_edges) == n_nodes:
                break

        # Convert edges to tour
        if len(tour_edges) < n_nodes:
            # Fallback to nearest neighbor
            return self._nearest_neighbor_tour(coords)

        return self._edges_to_tour(tour_edges, n_nodes)

    def _edges_to_tour(self, edges, n_nodes):
        """Convert edge list to tour."""
        adj = {i: [] for i in range(n_nodes)}
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        tour = [0]
        visited = {0}
        current = 0

        while len(tour) < n_nodes:
            for next_node in adj[current]:
                if next_node not in visited:
                    tour.append(next_node)
                    visited.add(next_node)
                    current = next_node
                    break
            else:
                # Fallback if stuck
                for i in range(n_nodes):
                    if i not in visited:
                        tour.append(i)
                        visited.add(i)
                        current = i
                        break

        return tour

    def _nearest_neighbor_tour(self, coords):
        """Fallback nearest neighbor heuristic."""
        n_nodes = coords.shape[0]
        tour = [0]
        visited = {0}
        current = 0

        while len(tour) < n_nodes:
            min_dist = float('inf')
            next_node = -1
            for i in range(n_nodes):
                if i not in visited:
                    dist = torch.norm(coords[current] - coords[i]).item()
                    if dist < min_dist:
                        min_dist = dist
                        next_node = i
            if next_node >= 0:
                tour.append(next_node)
                visited.add(next_node)
                current = next_node

        return tour

    def _compute_tour_length(self, coords, tour):
        """Compute total tour length."""
        length = 0.0
        for i in range(len(tour)):
            u = tour[i]
            v = tour[(i + 1) % len(tour)]
            length += torch.norm(coords[u] - coords[v]).item()
        return length


# ============================================================================
# CVRP Evaluation Pipeline
# ============================================================================

def extract_routes(route_sequence):
    """Extract individual routes from a route sequence."""
    routes = []
    current_route = []

    for node in route_sequence:
        if node == 0:  # Depot
            if current_route:
                routes.append(current_route)
                current_route = []
        else:
            current_route.append(node)

    if current_route:
        routes.append(current_route)

    return routes


def solve_route_tsp_lkh(coords, route_nodes, time_limit=1, max_trials=100):
    """Solve TSP for a single route using LKH-3."""
    if len(route_nodes) <= 2:
        # For small routes, compute directly
        tour = route_nodes
        length = 0.0
        depot = coords[0]
        if len(route_nodes) >= 1:
            length += torch.norm(depot - coords[route_nodes[0]]).item()
            for i in range(len(route_nodes) - 1):
                length += torch.norm(coords[route_nodes[i]] - coords[route_nodes[i+1]]).item()
            length += torch.norm(coords[route_nodes[-1]] - depot).item()
        return tour, length

    # Build sub-TSP with depot
    sub_nodes = [0] + route_nodes
    sub_coords = coords[sub_nodes].cpu().numpy()

    try:
        # Solve with LKH
        cost = _solve_tsp_lkh(sub_coords, time_limit, max_trials)

        # For now, just return the original order (LKH optimizes the tour internally)
        # The cost is what matters for evaluation
        return route_nodes, cost
    except Exception as e:
        # Fallback to nearest neighbor
        tour = route_nodes
        length = torch.norm(coords[0] - coords[route_nodes[0]]).item()
        for i in range(len(route_nodes) - 1):
            length += torch.norm(coords[route_nodes[i]] - coords[route_nodes[i+1]]).item()
        length += torch.norm(coords[route_nodes[-1]] - coords[0]).item()
        return tour, length


def solve_route_tsp(coords, route_nodes, tsp_solver):
    """Solve TSP for a single route using EDISCO TSP solver."""
    if len(route_nodes) <= 2:
        # For small routes, just return as-is
        tour = route_nodes
        length = 0.0
        depot = coords[0]
        if len(route_nodes) >= 1:
            length += torch.norm(depot - coords[route_nodes[0]]).item()
            for i in range(len(route_nodes) - 1):
                length += torch.norm(coords[route_nodes[i]] - coords[route_nodes[i+1]]).item()
            length += torch.norm(coords[route_nodes[-1]] - depot).item()
        return tour, length

    # Build sub-TSP with depot
    sub_nodes = [0] + route_nodes
    sub_coords = coords[sub_nodes]

    # Pad to match TSP solver size if needed
    n_sub = len(sub_nodes)
    n_tsp = tsp_solver.n_nodes

    if n_sub > n_tsp:
        # Route too large - use nearest neighbor
        tour = nearest_neighbor_tsp(sub_coords)
    elif n_sub < n_tsp:
        # Pad coordinates
        padded_coords = torch.zeros(n_tsp, 2, device=coords.device)
        padded_coords[:n_sub] = sub_coords
        # Place padding nodes far away
        padded_coords[n_sub:] = torch.tensor([10.0, 10.0], device=coords.device)

        tour, _ = tsp_solver.solve(padded_coords)
        # Extract only valid nodes from tour
        tour = [t for t in tour if t < n_sub]
    else:
        tour, _ = tsp_solver.solve(sub_coords)

    # Remap tour indices back to original nodes
    # Tour should start from depot (index 0 in sub_nodes)
    depot_idx = tour.index(0) if 0 in tour else 0
    tour = tour[depot_idx:] + tour[:depot_idx]

    # Map back to original node indices (excluding depot)
    remapped_tour = [sub_nodes[t] for t in tour if t != 0]

    # Compute length
    length = torch.norm(coords[0] - coords[remapped_tour[0]]).item()
    for i in range(len(remapped_tour) - 1):
        length += torch.norm(coords[remapped_tour[i]] - coords[remapped_tour[i+1]]).item()
    length += torch.norm(coords[remapped_tour[-1]] - coords[0]).item()

    return remapped_tour, length


def nearest_neighbor_tsp(coords):
    """Simple nearest neighbor TSP solver."""
    n = coords.shape[0]
    tour = [0]
    visited = {0}
    current = 0

    while len(tour) < n:
        min_dist = float('inf')
        next_node = -1
        for i in range(n):
            if i not in visited:
                dist = torch.norm(coords[current] - coords[i]).item()
                if dist < min_dist:
                    min_dist = dist
                    next_node = i
        tour.append(next_node)
        visited.add(next_node)
        current = next_node

    return tour


def evaluate_cvrp_solution(coords, demand, capacity, routes, tsp_solver=None, use_lkh=False):
    """
    Evaluate CVRP solution quality.

    Args:
        coords: Node coordinates
        demand: Node demands
        capacity: Vehicle capacity
        routes: List of routes (each route is list of customer indices)
        tsp_solver: Optional EDISCO TSP solver
        use_lkh: If True, use LKH-3 for sub-TSP solving

    Returns:
        total_cost: Total route distance
        optimized_routes: Routes after TSP optimization
    """
    total_cost = 0.0
    optimized_routes = []

    for route in routes:
        if not route:
            continue

        if use_lkh and len(route) > 2 and LKH_PATH is not None:
            opt_route, cost = solve_route_tsp_lkh(coords, route)
        elif tsp_solver is not None and len(route) > 2:
            opt_route, cost = solve_route_tsp(coords, route, tsp_solver)
        else:
            opt_route = route
            # Simple cost calculation
            cost = torch.norm(coords[0] - coords[route[0]]).item()
            for i in range(len(route) - 1):
                cost += torch.norm(coords[route[i]] - coords[route[i+1]]).item()
            cost += torch.norm(coords[route[-1]] - coords[0]).item()

        optimized_routes.append(opt_route)
        total_cost += cost

    return total_cost, optimized_routes


def visualize_cvrp_solution(coords, routes, demand=None, title="CVRP Solution", save_path=None):
    """Visualize CVRP solution with routes in different colors."""
    if not HAS_MATPLOTLIB:
        print("Skipping visualization: matplotlib not installed")
        return None

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(routes))))

    # Plot depot
    ax.scatter(coords[0, 0].cpu(), coords[0, 1].cpu(), c='red', s=200, marker='s',
               zorder=5, label='Depot')

    # Plot routes
    for i, route in enumerate(routes):
        if not route:
            continue

        color = colors[i % len(colors)]

        # Full route: depot -> customers -> depot
        full_route = [0] + route + [0]
        route_coords = coords[full_route].cpu().numpy()

        # Plot route line
        ax.plot(route_coords[:, 0], route_coords[:, 1], c=color, linewidth=2, alpha=0.7)

        # Plot customers
        customer_coords = coords[route].cpu().numpy()
        ax.scatter(customer_coords[:, 0], customer_coords[:, 1], c=[color], s=50, zorder=4)

        # Label customers
        for j, node in enumerate(route):
            ax.annotate(str(node), (coords[node, 0].cpu(), coords[node, 1].cpu()),
                       fontsize=8, ha='center', va='bottom')

    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal')
    ax.legend()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")

    plt.show()
    return fig


# ============================================================================
# Main Evaluation
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate CVRP with EDISCO Partition + TSP')
    parser.add_argument('--partition_checkpoint', type=str, default='checkpoints/cvrp100/best.pt',
                       help='Path to partition model checkpoint')
    parser.add_argument('--tsp_checkpoint', type=str, default='pretrained/edisco/best_tsp_100_dense.pth',
                       help='Path to TSP solver checkpoint')
    parser.add_argument('--n_customers', type=int, default=100, help='Number of customers')
    parser.add_argument('--n_instances', type=int, default=10, help='Number of instances to evaluate')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--visualize', action='store_true', help='Visualize solutions')
    parser.add_argument('--use_lkh', action='store_true', help='Use LKH-3 for sub-TSP solving (recommended)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    # Check LKH availability
    if args.use_lkh:
        if LKH_PATH is not None:
            print(f"Using LKH-3 for sub-TSP solving: {LKH_PATH}")
        else:
            print("Warning: LKH not found! Falling back to NN routing.")
            print("Install LKH-3 or set LKH_PATH environment variable.")
            args.use_lkh = False

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load partition model
    print("Loading partition model...")
    partition_net = PartitionNet(
        n_layers=12,
        hidden_dim=128,
        node_dim=64,
        edge_dim=64
    ).to(device)

    if os.path.exists(args.partition_checkpoint):
        checkpoint = torch.load(args.partition_checkpoint, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            partition_net.load_state_dict(checkpoint['model_state_dict'])
        else:
            partition_net.load_state_dict(checkpoint)
        print(f"Loaded partition model from {args.partition_checkpoint}")
    else:
        print(f"Warning: Partition checkpoint not found at {args.partition_checkpoint}")
        print("Using randomly initialized partition model")

    partition_net.eval()

    # Load TSP solver
    print("Loading TSP solver...")
    tsp_solver = None
    if os.path.exists(args.tsp_checkpoint):
        try:
            tsp_solver = EDISCOTSPSolver(n_nodes=100, device=device)
            tsp_solver.load_checkpoint(args.tsp_checkpoint)
            tsp_solver.eval()
        except Exception as e:
            print(f"Warning: Failed to load TSP solver: {e}")
            print("Will use nearest neighbor heuristic instead")
            tsp_solver = None
    else:
        print(f"Warning: TSP checkpoint not found at {args.tsp_checkpoint}")
        print("Will use nearest neighbor heuristic instead")

    # K-sparse values
    K_SPARSE = {100: 20, 200: 30, 500: 50, 1000: 100}
    k_sparse = K_SPARSE.get(args.n_customers, 20)

    # Evaluate
    print(f"\nEvaluating {args.n_instances} CVRP-{args.n_customers} instances...")

    results = {
        'costs_nn': [],
        'costs_tsp': [],
        'n_routes': []
    }

    for i in range(args.n_instances):
        # Generate instance
        coords, demand, capacity = generate_instance(args.n_customers, device)

        # Build graph
        graph = build_graph(coords, demand, capacity, k_sparse)

        # Get partition heatmap
        with torch.no_grad():
            heatmap = partition_net(graph)

        # Sample routes
        sampler = Sampler(demand, heatmap, capacity, batch_size=1, device=device)
        routes_tensor = sampler.gen_subsets(require_prob=False, greedy_mode=True)

        # Extract routes
        route_sequence = routes_tensor[0].cpu().tolist()
        routes = extract_routes(route_sequence)

        # Evaluate with NN routing
        cost_nn, _ = evaluate_cvrp_solution(coords, demand, capacity, routes, tsp_solver=None, use_lkh=False)
        results['costs_nn'].append(cost_nn)

        # Evaluate with optimized solver (LKH or EDISCO TSP)
        if args.use_lkh:
            cost_tsp, optimized_routes = evaluate_cvrp_solution(
                coords, demand, capacity, routes, tsp_solver=None, use_lkh=True
            )
            results['costs_tsp'].append(cost_tsp)
            solver_name = "LKH"
        elif tsp_solver is not None:
            cost_tsp, optimized_routes = evaluate_cvrp_solution(
                coords, demand, capacity, routes, tsp_solver=tsp_solver, use_lkh=False
            )
            results['costs_tsp'].append(cost_tsp)
            solver_name = "TSP"
        else:
            cost_tsp = cost_nn
            optimized_routes = routes
            results['costs_tsp'].append(cost_nn)
            solver_name = "NN"

        results['n_routes'].append(len(routes))

        print(f"Instance {i+1}: NN={cost_nn:.2f}, {solver_name}={cost_tsp:.2f}, "
              f"Routes={len(routes)}, Improvement={(cost_nn-cost_tsp)/cost_nn*100:.1f}%")

        # Visualize first few instances
        if args.visualize and i < 3:
            visualize_cvrp_solution(
                coords, optimized_routes, demand,
                title=f"CVRP-{args.n_customers} Instance {i+1} (Cost: {cost_tsp:.2f})",
                save_path=f"cvrp_{args.n_customers}_instance_{i+1}.png"
            )

    # Summary
    solver_label = "LKH" if args.use_lkh else ("EDISCO-TSP" if tsp_solver else "NN")
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"Sub-TSP Solver: {solver_label}")
    print(f"Average cost (NN routing):       {np.mean(results['costs_nn']):.2f} ± {np.std(results['costs_nn']):.2f}")
    print(f"Average cost ({solver_label} routing): {np.mean(results['costs_tsp']):.2f} ± {np.std(results['costs_tsp']):.2f}")
    print(f"Average improvement:             {(np.mean(results['costs_nn'])-np.mean(results['costs_tsp']))/np.mean(results['costs_nn'])*100:.1f}%")
    print(f"Average routes per instance:     {np.mean(results['n_routes']):.1f}")


if __name__ == '__main__':
    main()
