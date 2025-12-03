"""
LKH-3 Routing for CVRP Evaluation.

Higher quality than nearest neighbor, but slower.
Use for validation/testing, not training.
"""

import os
import subprocess
import tempfile
import shutil
import numpy as np
import torch


def find_lkh():
    """Find LKH-3 executable."""
    # Check environment variable
    env_path = os.environ.get('LKH_PATH')
    if env_path and os.path.isfile(env_path):
        return env_path

    # Check system PATH
    system_lkh = shutil.which('LKH')
    if system_lkh:
        return system_lkh

    # Common installation paths
    home = os.path.expanduser("~")
    common_paths = [
        f"{home}/LKH_install/LKH-3.0.8/LKH",
        f"{home}/LKH_install/LKH-3.0.6/LKH",
        f"{home}/Code/CVRP/LKH-3.0.8/LKH",
        f"{home}/LKH-3.0.8/LKH",
        f"{home}/LKH-3/LKH",
        "/usr/local/bin/LKH",
        "./LKH-3.0.8/LKH",
        "./LKH",
    ]

    for path in common_paths:
        if os.path.isfile(path):
            return path

    return None


# Global LKH path (set once at import)
LKH_PATH = find_lkh()


def eval_routes_lkh(coords, routes, capacity=None, demand=None,
                    time_limit=1, max_trials=100):
    """
    Evaluate routes using LKH-3 for each segment (TSP).

    Args:
        coords: (n_nodes, 2) - node coordinates
        routes: (batch_size, route_length) - sampled routes
        capacity: Not used, kept for API compatibility
        demand: Not used, kept for API compatibility
        time_limit: Seconds per segment
        max_trials: LKH trials per segment

    Returns:
        costs: (batch_size,) - total route distances
    """
    from .nn_routing import eval_routes_nn

    if LKH_PATH is None:
        print("Warning: LKH not found, falling back to NN routing")
        return eval_routes_nn(coords, routes)

    bs = routes.size(0)
    costs = []
    coords_np = coords.cpu().numpy()
    depot = coords_np[0]

    for i in range(bs):
        route = routes[i].cpu().numpy()
        cost = _evaluate_single_route_lkh(coords_np, depot, route, time_limit, max_trials)
        costs.append(cost)

    return torch.tensor(costs, device=coords.device, dtype=torch.float)


def _evaluate_single_route_lkh(coords_np, depot, route, time_limit, max_trials):
    """Evaluate a single route using LKH for each segment."""
    cost = 0.0

    # Split route at depot visits
    segments = []
    current_seg = []

    for node in route:
        if node == 0:
            if current_seg:
                segments.append(current_seg)
                current_seg = []
        else:
            current_seg.append(node)

    if current_seg:
        segments.append(current_seg)

    # Solve each segment
    for seg in segments:
        if not seg:
            continue

        n = len(seg)
        if n == 1:
            cost += 2 * np.linalg.norm(depot - coords_np[seg[0]])
        elif n == 2:
            cost += np.linalg.norm(depot - coords_np[seg[0]])
            cost += np.linalg.norm(coords_np[seg[0]] - coords_np[seg[1]])
            cost += np.linalg.norm(coords_np[seg[1]] - depot)
        else:
            # Build TSP instance with depot
            seg_with_depot = [0] + seg
            seg_coords = coords_np[seg_with_depot]

            try:
                seg_cost = _solve_tsp_lkh(seg_coords, time_limit, max_trials)
                cost += seg_cost
            except Exception:
                # Fallback to NN for this segment
                from .nn_routing import _nn_tsp
                cost += _nn_tsp(depot, coords_np[seg])

    return cost


def _solve_tsp_lkh(coords, time_limit=1, max_trials=100):
    """Solve TSP using LKH-3."""
    n = len(coords)
    scale = 100000  # Scale to integers for LKH

    with tempfile.TemporaryDirectory() as tmpdir:
        prob_file = os.path.join(tmpdir, "problem.tsp")
        par_file = os.path.join(tmpdir, "param.par")
        tour_file = os.path.join(tmpdir, "tour.txt")

        # Write TSP file
        with open(prob_file, 'w') as f:
            f.write(f"NAME : seg\n")
            f.write(f"TYPE : TSP\n")
            f.write(f"DIMENSION : {n}\n")
            f.write(f"EDGE_WEIGHT_TYPE : EUC_2D\n")
            f.write(f"NODE_COORD_SECTION\n")
            for i, (x, y) in enumerate(coords):
                f.write(f"{i+1} {int(x*scale)} {int(y*scale)}\n")
            f.write("EOF\n")

        # Write parameter file
        with open(par_file, 'w') as f:
            f.write(f"PROBLEM_FILE = {prob_file}\n")
            f.write(f"OUTPUT_TOUR_FILE = {tour_file}\n")
            f.write(f"TIME_LIMIT = {time_limit}\n")
            f.write(f"MAX_TRIALS = {max_trials}\n")
            f.write("RUNS = 1\n")
            f.write("SEED = 1\n")

        # Run LKH
        subprocess.run(
            [LKH_PATH, par_file],
            capture_output=True,
            timeout=time_limit + 5
        )

        # Parse tour
        with open(tour_file, 'r') as f:
            lines = f.readlines()

        tour = []
        in_tour = False
        for line in lines:
            if "TOUR_SECTION" in line:
                in_tour = True
                continue
            if in_tour:
                node = int(line.strip())
                if node == -1:
                    break
                tour.append(node - 1)  # Convert to 0-indexed

        # Compute tour length
        cost = 0.0
        for i in range(len(tour)):
            cost += np.linalg.norm(coords[tour[i]] - coords[tour[(i+1) % len(tour)]])

        return cost
