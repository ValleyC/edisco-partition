"""
Nearest Neighbor Routing for CVRP.

Fast evaluation method for training. Not optimal but sufficient
for relative ranking in REINFORCE.
"""

import torch
import numpy as np


def eval_routes_nn(coords, routes):
    """
    Evaluate routes using nearest neighbor TSP within each segment.

    Args:
        coords: (n_nodes, 2) - node coordinates
        routes: (batch_size, route_length) - sampled routes

    Returns:
        costs: (batch_size,) - total route distances
    """
    bs = routes.size(0)
    costs = []
    coords_np = coords.cpu().numpy()
    depot = coords_np[0]

    for i in range(bs):
        route = routes[i].cpu().numpy()
        cost = _evaluate_single_route(coords_np, depot, route)
        costs.append(cost)

    return torch.tensor(costs, device=coords.device, dtype=torch.float)


def _evaluate_single_route(coords_np, depot, route):
    """Evaluate a single route."""
    cost = 0.0

    # Split route at depot visits to get segments
    segments = []
    current_seg = []

    for node in route:
        if node == 0:  # Depot
            if current_seg:
                segments.append(current_seg)
                current_seg = []
        else:
            current_seg.append(node)

    if current_seg:
        segments.append(current_seg)

    # Solve each segment with nearest neighbor
    for seg in segments:
        if not seg:
            continue

        n = len(seg)
        if n == 1:
            # Single customer: depot -> customer -> depot
            cost += 2 * np.linalg.norm(depot - coords_np[seg[0]])
        elif n == 2:
            # Two customers: depot -> c1 -> c2 -> depot
            cost += np.linalg.norm(depot - coords_np[seg[0]])
            cost += np.linalg.norm(coords_np[seg[0]] - coords_np[seg[1]])
            cost += np.linalg.norm(coords_np[seg[1]] - depot)
        else:
            # Multiple customers: nearest neighbor TSP
            seg_coords = coords_np[seg]
            seg_cost = _nn_tsp(depot, seg_coords)
            cost += seg_cost

    return cost


def _nn_tsp(depot, seg_coords):
    """
    Solve TSP on segment using nearest neighbor heuristic.

    Args:
        depot: (2,) - depot coordinates
        seg_coords: (n, 2) - customer coordinates in this segment

    Returns:
        cost: Total tour length (depot -> customers -> depot)
    """
    n = len(seg_coords)
    visited = [False] * n
    current = depot
    cost = 0.0

    for _ in range(n):
        # Find nearest unvisited customer
        best_dist = float('inf')
        best_idx = -1

        for j in range(n):
            if not visited[j]:
                dist = np.linalg.norm(current - seg_coords[j])
                if dist < best_dist:
                    best_dist = dist
                    best_idx = j

        if best_idx >= 0:
            visited[best_idx] = True
            cost += best_dist
            current = seg_coords[best_idx]

    # Return to depot
    cost += np.linalg.norm(current - depot)

    return cost


def eval_routes_nn_batch(coords, routes):
    """
    Batch-friendly version (same as eval_routes_nn but clearer name).
    """
    return eval_routes_nn(coords, routes)
