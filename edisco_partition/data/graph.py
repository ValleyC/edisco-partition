"""
Graph Construction for E(2)-Equivariant CVRP Partition.

Key difference from GLOP: We do NOT use polar angle theta as a feature,
as it breaks rotation equivariance.

Node features (invariant):
- demand / capacity
- distance from depot

Edge features (invariant):
- Euclidean distance
- Affinity (1 - normalized distance)
"""

import torch
from dataclasses import dataclass
from typing import Optional


@dataclass
class GraphData:
    """Simple graph data container."""
    x: torch.Tensor           # Node features (n_nodes, node_dim)
    pos: torch.Tensor         # Node coordinates (n_nodes, 2)
    edge_index: torch.Tensor  # Edge indices (2, n_edges)
    edge_attr: torch.Tensor   # Edge features (n_edges, edge_dim)
    n_nodes: int
    k_sparse: int


def build_graph(coords, demand, capacity, k_sparse):
    """
    Build a k-sparse graph for CVRP partition.

    Args:
        coords: (n_nodes, 2) - coordinates with depot at index 0
        demand: (n_nodes,) - demands with depot = 0
        capacity: Vehicle capacity
        k_sparse: Number of nearest neighbors per node

    Returns:
        GraphData with E(2)-equivariant features
    """
    n_nodes = coords.shape[0]
    device = coords.device

    # =========================================================================
    # Node Features (INVARIANT - no polar angle!)
    # =========================================================================

    # Normalized demand
    norm_demand = demand / capacity

    # Distance from depot (rotation invariant)
    depot = coords[0:1]  # (1, 2)
    shift_coords = coords - depot  # Shift so depot is at origin
    dist_from_depot = torch.norm(shift_coords, dim=1)  # (n_nodes,)

    # Stack node features: (demand/capacity, distance_from_depot)
    # Note: NO theta (polar angle) - this is the key difference from GLOP
    x = torch.stack([norm_demand, dist_from_depot], dim=1)  # (n_nodes, 2)

    # =========================================================================
    # Edge Construction (k-sparse based on distance)
    # =========================================================================

    # Compute full distance matrix
    dist_mat = torch.cdist(coords, coords)  # (n_nodes, n_nodes)

    # For each node, find k nearest neighbors
    # Use negative distance so topk gives smallest distances
    _, topk_indices = torch.topk(-dist_mat, k=k_sparse, dim=1)

    # Build edge_index
    row = torch.arange(n_nodes, device=device).unsqueeze(1).expand(-1, k_sparse).flatten()
    col = topk_indices.flatten()
    edge_index = torch.stack([row, col], dim=0)  # (2, n_nodes * k_sparse)

    # =========================================================================
    # Edge Features (INVARIANT)
    # =========================================================================

    # Edge distances
    edge_dist = dist_mat[edge_index[0], edge_index[1]]  # (n_edges,)

    # Edge affinity (1 - normalized distance)
    max_dist = dist_mat.max() + 1e-8
    edge_affinity = 1 - edge_dist / max_dist

    # Stack edge features: (distance, affinity)
    edge_attr = torch.stack([edge_dist, edge_affinity], dim=1)  # (n_edges, 2)

    return GraphData(
        x=x,
        pos=coords,
        edge_index=edge_index,
        edge_attr=edge_attr,
        n_nodes=n_nodes,
        k_sparse=k_sparse
    )


def build_graph_cosine(coords, demand, capacity, k_sparse):
    """
    Build k-sparse graph using cosine similarity (like GLOP).

    This version uses cosine similarity for neighbor selection,
    which tends to group nodes in similar angular directions.
    Still E(2)-equivariant because we don't use theta as a feature.

    Args:
        coords: (n_nodes, 2) - coordinates with depot at index 0
        demand: (n_nodes,) - demands with depot = 0
        capacity: Vehicle capacity
        k_sparse: Number of nearest neighbors per node

    Returns:
        GraphData with E(2)-equivariant features
    """
    n_nodes = coords.shape[0]
    device = coords.device

    # Node features (invariant)
    norm_demand = demand / capacity
    depot = coords[0:1]
    shift_coords = coords - depot
    dist_from_depot = torch.norm(shift_coords, dim=1)
    x = torch.stack([norm_demand, dist_from_depot], dim=1)

    # Cosine similarity for neighbor selection
    # cos(i, j) = (shift_i . shift_j) / (||shift_i|| * ||shift_j||)
    dot_products = torch.mm(shift_coords, shift_coords.t())
    magnitudes = dist_from_depot.unsqueeze(1)
    magnitude_matrix = magnitudes * magnitudes.t() + 1e-10
    cos_sim = dot_products / magnitude_matrix

    # Also consider Euclidean affinity
    dist_mat = torch.cdist(coords, coords)
    max_dist = dist_mat.max() + 1e-8
    euc_affinity = 1 - dist_mat / max_dist

    # Combined score for neighbor selection
    combined_score = cos_sim + euc_affinity

    # Select top-k neighbors
    _, topk_indices = torch.topk(combined_score, k=k_sparse, dim=1)

    # Build edge_index
    row = torch.arange(n_nodes, device=device).unsqueeze(1).expand(-1, k_sparse).flatten()
    col = topk_indices.flatten()
    edge_index = torch.stack([row, col], dim=0)

    # Edge features (invariant)
    edge_dist = dist_mat[edge_index[0], edge_index[1]]
    edge_affinity = 1 - edge_dist / max_dist
    edge_attr = torch.stack([edge_dist, edge_affinity], dim=1)

    return GraphData(
        x=x,
        pos=coords,
        edge_index=edge_index,
        edge_attr=edge_attr,
        n_nodes=n_nodes,
        k_sparse=k_sparse
    )
