"""
Random insertion heuristic for TSP from GLOP.

Uses the random_insertion package if available, otherwise falls back to identity order.
"""
import torch
import numpy as np

# Try to import random_insertion, fall back gracefully
try:
    import random_insertion as ri
    HAS_RANDOM_INSERTION = True
except ImportError:
    HAS_RANDOM_INSERTION = False
    print("Warning: random-insertion package not found. Install with: pip install random-insertion")


def _to_numpy(arr):
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    if isinstance(arr, list):
        return np.array(arr)
    else:
        return arr


def random_insertion(cities, order=None):
    """Single instance random insertion."""
    if not HAS_RANDOM_INSERTION:
        # Fallback: return identity order
        return np.arange(len(cities))
    cities = _to_numpy(cities)
    order = _to_numpy(order)
    return ri.tsp_random_insertion(cities, order)


def random_insertion_parallel(cities, orders):
    """Parallel batch random insertion."""
    if not HAS_RANDOM_INSERTION:
        # Fallback: return identity orders
        batch_size = cities.shape[0]
        n_nodes = cities.shape[1]
        return np.tile(np.arange(n_nodes), (batch_size, 1))
    cities = _to_numpy(cities)
    orders = _to_numpy(orders)
    return ri.tsp_random_insertion_parallel(cities, orders)


def random_insertion_non_euclidean(distmap, order):
    """Random insertion for non-Euclidean (asymmetric) TSP."""
    if not HAS_RANDOM_INSERTION:
        return np.arange(distmap.shape[0])
    distmap = _to_numpy(distmap)
    order = _to_numpy(order)
    return ri.atsp_random_insertion(distmap, order)
