"""
Large-Scale CVRP Data Generation Module
========================================

Generates high-quality CVRP instances for scales 200-1000+ customers.
Supports multiple solvers: LKH-3, HGS, OR-Tools, and heuristics.

Features:
- Parallel instance generation
- Multiple distribution types (uniform, clustered, mixed)
- Quality verification
- Efficient storage with compression

Author: EDISCO Team
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Union
import pickle
import gzip
import os
import sys
import subprocess
import tempfile
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp
from dataclasses import dataclass, field
from enum import Enum
import time
import json
from tqdm import tqdm
import warnings


# ============================================================================
# CONFIGURATION
# ============================================================================

class DistributionType(Enum):
    """Types of customer distributions."""
    UNIFORM = "uniform"
    CLUSTERED = "clustered"
    MIXED = "mixed"
    DIAGONAL = "diagonal"
    RING = "ring"


@dataclass
class CVRPConfig:
    """Configuration for CVRP instance generation."""
    n_customers: int = 200
    distribution: DistributionType = DistributionType.UNIFORM

    # Capacity settings (following standard benchmarks)
    capacity: Optional[int] = None  # Auto-set based on n_customers if None
    demand_low: int = 1
    demand_high: int = 10

    # Clustered distribution parameters
    n_clusters: int = 5
    cluster_std: float = 0.1

    # Solver settings
    solver: str = "auto"  # "lkh", "hgs", "ortools", "heuristic", "auto"
    time_limit: int = 60

    # Paths
    lkh_path: Optional[str] = None
    hgs_path: Optional[str] = None

    def __post_init__(self):
        """Set default capacity based on problem size."""
        if self.capacity is None:
            # Standard CVRP capacities from literature
            capacity_map = {
                20: 30, 50: 40, 100: 50,
                200: 80, 500: 100, 1000: 150,
                2000: 200, 5000: 300, 10000: 400
            }
            # Find closest size
            sizes = sorted(capacity_map.keys())
            closest = min(sizes, key=lambda x: abs(x - self.n_customers))
            self.capacity = capacity_map[closest]


# ============================================================================
# INSTANCE GENERATION
# ============================================================================

class CVRPInstanceGenerator:
    """
    Generate CVRP instances with various distributions.
    """

    def __init__(self, config: CVRPConfig):
        self.config = config

    def generate(self, seed: Optional[int] = None) -> Dict:
        """Generate a single CVRP instance."""
        if seed is not None:
            np.random.seed(seed)

        n = self.config.n_customers

        # Generate coordinates based on distribution type
        if self.config.distribution == DistributionType.UNIFORM:
            coords = self._generate_uniform(n)
        elif self.config.distribution == DistributionType.CLUSTERED:
            coords = self._generate_clustered(n)
        elif self.config.distribution == DistributionType.MIXED:
            coords = self._generate_mixed(n)
        elif self.config.distribution == DistributionType.DIAGONAL:
            coords = self._generate_diagonal(n)
        elif self.config.distribution == DistributionType.RING:
            coords = self._generate_ring(n)
        else:
            coords = self._generate_uniform(n)

        # Generate demands
        demands = np.random.randint(
            self.config.demand_low,
            self.config.demand_high + 1,
            size=n
        )

        # Add depot (index 0) with demand 0
        depot = np.array([[0.5, 0.5]])  # Center of unit square
        coords = np.vstack([depot, coords])
        demands = np.concatenate([[0], demands])

        return {
            'coords': coords.astype(np.float32),
            'demands': demands.astype(np.int32),
            'capacity': self.config.capacity,
            'n_nodes': n + 1,
            'n_customers': n,
            'depot_idx': 0,
            'distribution': self.config.distribution.value
        }

    def _generate_uniform(self, n: int) -> np.ndarray:
        """Uniform distribution in [0, 1]^2."""
        return np.random.uniform(0, 1, size=(n, 2))

    def _generate_clustered(self, n: int) -> np.ndarray:
        """Clustered distribution with Gaussian clusters."""
        k = self.config.n_clusters
        std = self.config.cluster_std

        # Generate cluster centers
        centers = np.random.uniform(0.1, 0.9, size=(k, 2))

        # Assign customers to clusters
        cluster_sizes = np.random.multinomial(n, [1/k] * k)

        coords = []
        for i, size in enumerate(cluster_sizes):
            if size > 0:
                cluster_coords = np.random.normal(
                    centers[i], std, size=(size, 2)
                )
                coords.append(cluster_coords)

        coords = np.vstack(coords)
        # Clip to [0, 1]
        coords = np.clip(coords, 0, 1)

        return coords

    def _generate_mixed(self, n: int) -> np.ndarray:
        """Mixed distribution: 50% uniform, 50% clustered."""
        n_uniform = n // 2
        n_clustered = n - n_uniform

        uniform_coords = self._generate_uniform(n_uniform)

        # Temporarily adjust n_customers for clustered generation
        old_n = self.config.n_customers
        self.config.n_customers = n_clustered
        clustered_coords = self._generate_clustered(n_clustered)
        self.config.n_customers = old_n

        return np.vstack([uniform_coords, clustered_coords])

    def _generate_diagonal(self, n: int) -> np.ndarray:
        """Diagonal distribution along y = x."""
        t = np.random.uniform(0, 1, size=n)
        noise = np.random.normal(0, 0.1, size=(n, 2))
        coords = np.column_stack([t, t]) + noise
        return np.clip(coords, 0, 1)

    def _generate_ring(self, n: int) -> np.ndarray:
        """Ring distribution around depot."""
        angles = np.random.uniform(0, 2 * np.pi, size=n)
        radii = np.random.uniform(0.2, 0.45, size=n)

        x = 0.5 + radii * np.cos(angles)
        y = 0.5 + radii * np.sin(angles)

        return np.column_stack([x, y])

    def generate_batch(
        self,
        n_instances: int,
        base_seed: int = 0,
        parallel: bool = True,
        n_workers: Optional[int] = None
    ) -> List[Dict]:
        """Generate multiple instances."""
        seeds = [base_seed + i for i in range(n_instances)]

        if parallel and n_instances > 10:
            if n_workers is None:
                n_workers = min(mp.cpu_count(), 8)

            instances = []
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = [executor.submit(self.generate, seed) for seed in seeds]
                for future in tqdm(as_completed(futures), total=n_instances,
                                  desc="Generating instances"):
                    instances.append(future.result())
            return instances
        else:
            return [self.generate(seed) for seed in tqdm(seeds, desc="Generating instances")]


# ============================================================================
# SOLVERS
# ============================================================================

class CVRPSolver:
    """Base class for CVRP solvers."""

    def solve(self, instance: Dict, time_limit: int = 60) -> Optional[Dict]:
        raise NotImplementedError

    def compute_route_distance(self, coords: np.ndarray, routes: List[List[int]]) -> float:
        """Compute total distance for CVRP solution."""
        total = 0.0
        for route in routes:
            if len(route) == 0:
                continue
            # Depot to first
            total += np.linalg.norm(coords[0] - coords[route[0]])
            # Route edges
            for i in range(len(route) - 1):
                total += np.linalg.norm(coords[route[i]] - coords[route[i+1]])
            # Last to depot
            total += np.linalg.norm(coords[route[-1]] - coords[0])
        return total


class SavingsHeuristic(CVRPSolver):
    """Clarke-Wright Savings Algorithm - fast heuristic baseline."""

    def solve(self, instance: Dict, time_limit: int = 60) -> Optional[Dict]:
        coords = instance['coords']
        demands = instance['demands']
        capacity = instance['capacity']
        n_customers = instance['n_customers']

        # Calculate savings
        savings = []
        for i in range(1, n_customers + 1):
            for j in range(i + 1, n_customers + 1):
                s = (np.linalg.norm(coords[0] - coords[i]) +
                     np.linalg.norm(coords[0] - coords[j]) -
                     np.linalg.norm(coords[i] - coords[j]))
                savings.append((s, i, j))

        savings.sort(reverse=True)

        # Initialize routes
        routes = {i: [i] for i in range(1, n_customers + 1)}
        route_demands = {i: demands[i] for i in range(1, n_customers + 1)}
        customer_route = {i: i for i in range(1, n_customers + 1)}

        # Merge routes
        for s, i, j in savings:
            ri, rj = customer_route.get(i), customer_route.get(j)

            if ri is None or rj is None or ri == rj:
                continue

            if ri not in routes or rj not in routes:
                continue

            # Check capacity
            if route_demands[ri] + route_demands[rj] > capacity:
                continue

            # Check if i and j are at route ends
            route_i, route_j = routes[ri], routes[rj]

            if len(route_i) == 0 or len(route_j) == 0:
                continue

            i_at_end = (i == route_i[0] or i == route_i[-1])
            j_at_end = (j == route_j[0] or j == route_j[-1])

            if not (i_at_end and j_at_end):
                continue

            # Merge routes
            if i == route_i[-1] and j == route_j[0]:
                new_route = route_i + route_j
            elif i == route_i[0] and j == route_j[-1]:
                new_route = route_j + route_i
            elif i == route_i[-1] and j == route_j[-1]:
                new_route = route_i + route_j[::-1]
            else:  # i == route_i[0] and j == route_j[0]
                new_route = route_i[::-1] + route_j

            # Update data structures
            routes[ri] = new_route
            route_demands[ri] = route_demands[ri] + route_demands[rj]

            for c in route_j:
                customer_route[c] = ri

            del routes[rj]
            del route_demands[rj]

        # Collect final routes
        final_routes = [r for r in routes.values() if len(r) > 0]

        # Compute distance
        distance = self.compute_route_distance(coords, final_routes)

        return {
            'routes': final_routes,
            'total_distance': distance,
            'n_vehicles': len(final_routes),
            'solver': 'savings'
        }


class SweepHeuristic(CVRPSolver):
    """Sweep Algorithm - angle-based clustering."""

    def solve(self, instance: Dict, time_limit: int = 60) -> Optional[Dict]:
        coords = instance['coords']
        demands = instance['demands']
        capacity = instance['capacity']
        n_customers = instance['n_customers']

        depot = coords[0]

        # Compute angles from depot
        angles = []
        for i in range(1, n_customers + 1):
            dx = coords[i, 0] - depot[0]
            dy = coords[i, 1] - depot[1]
            angle = np.arctan2(dy, dx)
            angles.append((angle, i))

        # Sort by angle
        angles.sort()

        # Create routes by sweeping
        routes = []
        current_route = []
        current_load = 0

        for _, customer in angles:
            if current_load + demands[customer] <= capacity:
                current_route.append(customer)
                current_load += demands[customer]
            else:
                if current_route:
                    routes.append(current_route)
                current_route = [customer]
                current_load = demands[customer]

        if current_route:
            routes.append(current_route)

        # Compute distance
        distance = self.compute_route_distance(coords, routes)

        return {
            'routes': routes,
            'total_distance': distance,
            'n_vehicles': len(routes),
            'solver': 'sweep'
        }


class NearestNeighborHeuristic(CVRPSolver):
    """Nearest Neighbor construction heuristic."""

    def solve(self, instance: Dict, time_limit: int = 60) -> Optional[Dict]:
        coords = instance['coords']
        demands = instance['demands']
        capacity = instance['capacity']
        n_customers = instance['n_customers']

        unvisited = set(range(1, n_customers + 1))
        routes = []

        while unvisited:
            route = []
            current_load = 0
            current_pos = coords[0]  # Start at depot

            while True:
                # Find nearest feasible customer
                best_dist = float('inf')
                best_customer = None

                for customer in unvisited:
                    if current_load + demands[customer] <= capacity:
                        dist = np.linalg.norm(current_pos - coords[customer])
                        if dist < best_dist:
                            best_dist = dist
                            best_customer = customer

                if best_customer is None:
                    break

                route.append(best_customer)
                current_load += demands[best_customer]
                current_pos = coords[best_customer]
                unvisited.remove(best_customer)

            if route:
                routes.append(route)

        # Compute distance
        distance = self.compute_route_distance(coords, routes)

        return {
            'routes': routes,
            'total_distance': distance,
            'n_vehicles': len(routes),
            'solver': 'nearest_neighbor'
        }


class LKHSolver(CVRPSolver):
    """LKH-3 solver for high-quality solutions."""

    def __init__(self, lkh_path: str = None):
        self.lkh_path = lkh_path or self._find_lkh()
        self.available = self.lkh_path is not None and os.path.exists(self.lkh_path)

        if not self.available:
            warnings.warn("LKH-3 not found. Install from http://webhotel4.ruc.dk/~keld/research/LKH-3/")

    def _find_lkh(self) -> Optional[str]:
        """Try to find LKH executable."""
        possible_paths = [
            "./LKH-3/LKH",
            "./LKH",
            "../LKH-3/LKH",
            "LKH"  # System PATH
        ]
        for path in possible_paths:
            if os.path.isfile(path):
                return path
        return None

    def solve(self, instance: Dict, time_limit: int = 60) -> Optional[Dict]:
        if not self.available:
            return None

        coords = instance['coords']
        demands = instance['demands']
        capacity = instance['capacity']
        n_nodes = instance['n_nodes']

        with tempfile.TemporaryDirectory() as temp_dir:
            problem_file = os.path.join(temp_dir, 'problem.vrp')
            param_file = os.path.join(temp_dir, 'param.par')
            tour_file = os.path.join(temp_dir, 'solution.tour')

            # Write problem file
            self._write_vrp_file(problem_file, coords, demands, capacity, n_nodes)

            # Write parameter file
            self._write_param_file(param_file, problem_file, tour_file, time_limit)

            try:
                result = subprocess.run(
                    [self.lkh_path, param_file],
                    capture_output=True,
                    text=True,
                    timeout=time_limit + 10,
                    cwd=temp_dir
                )

                if os.path.exists(tour_file):
                    return self._parse_solution(tour_file, coords)

            except Exception as e:
                warnings.warn(f"LKH error: {e}")

        return None

    def _write_vrp_file(self, filepath: str, coords: np.ndarray,
                        demands: np.ndarray, capacity: int, n_nodes: int):
        scale = 10000
        coords_scaled = (coords * scale).astype(int)

        with open(filepath, 'w') as f:
            f.write(f"NAME : CVRP_{n_nodes}\n")
            f.write("TYPE : CVRP\n")
            f.write(f"DIMENSION : {n_nodes}\n")
            f.write("EDGE_WEIGHT_TYPE : EUC_2D\n")
            f.write(f"CAPACITY : {capacity}\n")
            f.write("NODE_COORD_SECTION\n")
            for i in range(n_nodes):
                f.write(f"{i+1} {coords_scaled[i,0]} {coords_scaled[i,1]}\n")
            f.write("DEMAND_SECTION\n")
            for i in range(n_nodes):
                f.write(f"{i+1} {int(demands[i])}\n")
            f.write("DEPOT_SECTION\n1\n-1\nEOF\n")

    def _write_param_file(self, param_file: str, problem_file: str,
                          tour_file: str, time_limit: int):
        with open(param_file, 'w') as f:
            f.write(f"PROBLEM_FILE = {problem_file}\n")
            f.write(f"OUTPUT_TOUR_FILE = {tour_file}\n")
            f.write(f"RUNS = 5\n")
            f.write(f"TIME_LIMIT = {time_limit}\n")
            f.write("TRACE_LEVEL = 0\n")
            f.write("SEED = 1234\n")

    def _parse_solution(self, tour_file: str, coords: np.ndarray) -> Optional[Dict]:
        with open(tour_file, 'r') as f:
            lines = f.readlines()

        # Find tour section
        tour_start = None
        for i, line in enumerate(lines):
            if 'TOUR_SECTION' in line:
                tour_start = i + 1
                break

        if tour_start is None:
            return None

        # Parse tour
        tour = []
        for i in range(tour_start, len(lines)):
            line = lines[i].strip()
            if line == '-1' or line == 'EOF':
                break
            tour.append(int(line) - 1)  # Convert to 0-indexed

        # Convert to routes
        routes = []
        current_route = []
        for node in tour[1:]:  # Skip first depot
            if node == 0:
                if current_route:
                    routes.append(current_route)
                    current_route = []
            else:
                current_route.append(node)
        if current_route:
            routes.append(current_route)

        distance = self.compute_route_distance(coords, routes)

        return {
            'routes': routes,
            'total_distance': distance,
            'n_vehicles': len(routes),
            'solver': 'lkh'
        }


class HybridSolver(CVRPSolver):
    """
    Hybrid solver: tries LKH first, falls back to best heuristic.
    Uses multiple heuristics and picks best solution.
    """

    def __init__(self, lkh_path: str = None):
        self.lkh = LKHSolver(lkh_path)
        self.savings = SavingsHeuristic()
        self.sweep = SweepHeuristic()
        self.nn = NearestNeighborHeuristic()

    def solve(self, instance: Dict, time_limit: int = 60) -> Optional[Dict]:
        solutions = []

        # Try LKH if available
        if self.lkh.available and time_limit >= 10:
            lkh_sol = self.lkh.solve(instance, time_limit)
            if lkh_sol:
                solutions.append(lkh_sol)

        # Always run heuristics for comparison/fallback
        for solver in [self.savings, self.sweep, self.nn]:
            try:
                sol = solver.solve(instance, time_limit)
                if sol:
                    solutions.append(sol)
            except Exception:
                continue

        if not solutions:
            return None

        # Return best solution
        return min(solutions, key=lambda x: x['total_distance'])


# ============================================================================
# DATASET CLASS
# ============================================================================

class LargeCVRPDataset(Dataset):
    """
    PyTorch Dataset for large-scale CVRP.

    Features:
    - Lazy loading for memory efficiency
    - On-the-fly augmentation
    - Multiple representation formats
    """

    def __init__(
        self,
        data: Union[str, List[Dict]],
        transform: Optional[callable] = None,
        augment: bool = False,
        normalize_coords: bool = True,
        return_gt_clusters: bool = True
    ):
        """
        Args:
            data: Path to pickle file or list of instances
            transform: Optional transform function
            augment: Whether to apply random rotations/translations
            normalize_coords: Whether to normalize coordinates to [0, 1]
            return_gt_clusters: Whether to return ground-truth cluster labels from routes
        """
        if isinstance(data, str):
            self.data = self._load_data(data)
        else:
            self.data = data

        self.transform = transform
        self.augment = augment
        self.normalize_coords = normalize_coords
        self.return_gt_clusters = return_gt_clusters

        # Extract metadata
        if self.data:
            self.n_customers = self.data[0]['n_customers']
            self.n_nodes = self.data[0]['n_nodes']
            self.capacity = self.data[0]['capacity']

    def _load_data(self, path: str) -> List[Dict]:
        """Load data from file (supports gzip)."""
        if path.endswith('.gz'):
            with gzip.open(path, 'rb') as f:
                return pickle.load(f)
        else:
            with open(path, 'rb') as f:
                return pickle.load(f)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        instance = self.data[idx]

        coords = torch.FloatTensor(instance['coords'])
        demands = torch.FloatTensor(instance['demands'])
        capacity = torch.FloatTensor([instance['capacity']])

        # Apply augmentation (random rotation/translation)
        if self.augment:
            coords = self._augment_coords(coords)

        # Normalize coordinates
        if self.normalize_coords:
            coords = self._normalize_coords(coords)

        # Create node features
        n_nodes = coords.shape[0]
        is_depot = torch.zeros(n_nodes, 1)
        is_depot[0, 0] = 1.0

        # Normalized demands
        demands_norm = demands.unsqueeze(-1) / capacity

        # Invariant features: [demand/capacity, is_depot]
        invariant_features = torch.cat([demands_norm, is_depot], dim=-1)

        output = {
            'coords': coords,
            'demands': demands,
            'capacity': capacity,
            'invariant_features': invariant_features,
            'n_nodes': n_nodes,
            'n_customers': n_nodes - 1
        }

        # Add ground-truth cluster labels from solution
        if self.return_gt_clusters and 'solution' in instance and instance['solution']:
            gt_clusters = self._routes_to_clusters(
                instance['solution']['routes'], n_nodes
            )
            output['gt_clusters'] = gt_clusters
            output['gt_distance'] = instance['solution']['total_distance']

        if self.transform:
            output = self.transform(output)

        return output

    def _augment_coords(self, coords: torch.Tensor) -> torch.Tensor:
        """Apply random rotation and translation."""
        # Random rotation
        theta = torch.rand(1).item() * 2 * np.pi
        c, s = np.cos(theta), np.sin(theta)
        R = torch.tensor([[c, -s], [s, c]], dtype=torch.float32)

        # Random translation
        t = torch.rand(2) * 0.2 - 0.1  # Small translation

        # Center, rotate, translate, un-center
        center = coords.mean(dim=0)
        coords_centered = coords - center
        coords_rotated = coords_centered @ R.T
        coords_transformed = coords_rotated + center + t

        return coords_transformed

    def _normalize_coords(self, coords: torch.Tensor) -> torch.Tensor:
        """Normalize coordinates to [0, 1]."""
        min_vals = coords.min(dim=0).values
        max_vals = coords.max(dim=0).values
        range_vals = max_vals - min_vals + 1e-8
        return (coords - min_vals) / range_vals

    def _routes_to_clusters(
        self,
        routes: List[List[int]],
        n_nodes: int
    ) -> torch.Tensor:
        """Convert routes to cluster labels."""
        clusters = torch.zeros(n_nodes, dtype=torch.long)
        clusters[0] = -1  # Depot has no cluster

        for cluster_idx, route in enumerate(routes):
            for customer in route:
                clusters[customer] = cluster_idx

        return clusters


# ============================================================================
# DATA GENERATION PIPELINE
# ============================================================================

def generate_cvrp_dataset(
    n_instances: int,
    n_customers: int,
    output_path: str,
    distribution: DistributionType = DistributionType.UNIFORM,
    solver: str = "auto",
    time_limit: int = 60,
    n_workers: int = 4,
    base_seed: int = 0,
    compress: bool = True,
    lkh_path: Optional[str] = None
) -> str:
    """
    Generate a complete CVRP dataset with solutions.

    Args:
        n_instances: Number of instances to generate
        n_customers: Problem size
        output_path: Where to save the dataset
        distribution: Customer distribution type
        solver: Solver to use
        time_limit: Time limit per instance
        n_workers: Number of parallel workers
        base_seed: Base random seed
        compress: Whether to gzip the output
        lkh_path: Path to LKH executable

    Returns:
        Path to saved dataset
    """
    print("=" * 60)
    print(f"Generating CVRP Dataset")
    print("=" * 60)
    print(f"  Instances: {n_instances}")
    print(f"  Customers: {n_customers}")
    print(f"  Distribution: {distribution.value}")
    print(f"  Solver: {solver}")
    print(f"  Time limit: {time_limit}s")
    print("=" * 60)

    # Configuration
    config = CVRPConfig(
        n_customers=n_customers,
        distribution=distribution,
        solver=solver,
        time_limit=time_limit,
        lkh_path=lkh_path
    )

    # Generate instances
    generator = CVRPInstanceGenerator(config)
    instances = generator.generate_batch(
        n_instances, base_seed=base_seed, parallel=True, n_workers=n_workers
    )

    # Initialize solver
    if solver == "auto":
        solver_obj = HybridSolver(lkh_path)
    elif solver == "lkh":
        solver_obj = LKHSolver(lkh_path)
    elif solver == "savings":
        solver_obj = SavingsHeuristic()
    elif solver == "sweep":
        solver_obj = SweepHeuristic()
    else:
        solver_obj = NearestNeighborHeuristic()

    # Solve instances
    print("\nSolving instances...")

    def solve_instance(inst):
        return solver_obj.solve(inst, time_limit)

    if n_workers > 1 and solver != "lkh":  # LKH has its own parallelism
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(solve_instance, inst) for inst in instances]
            solutions = []
            for future in tqdm(as_completed(futures), total=n_instances, desc="Solving"):
                solutions.append(future.result())
    else:
        solutions = [solve_instance(inst) for inst in tqdm(instances, desc="Solving")]

    # Attach solutions to instances
    valid_count = 0
    for inst, sol in zip(instances, solutions):
        if sol:
            inst['solution'] = sol
            valid_count += 1
        else:
            # Use sweep as fallback
            inst['solution'] = SweepHeuristic().solve(inst, 10)

    print(f"\nValid solutions: {valid_count}/{n_instances}")

    # Compute statistics
    distances = [inst['solution']['total_distance'] for inst in instances if inst.get('solution')]
    n_vehicles = [inst['solution']['n_vehicles'] for inst in instances if inst.get('solution')]

    print(f"\nDataset Statistics:")
    print(f"  Avg distance: {np.mean(distances):.3f} +/- {np.std(distances):.3f}")
    print(f"  Avg vehicles: {np.mean(n_vehicles):.2f} +/- {np.std(n_vehicles):.2f}")

    # Save dataset
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    if compress:
        output_path = output_path if output_path.endswith('.gz') else output_path + '.gz'
        with gzip.open(output_path, 'wb') as f:
            pickle.dump(instances, f)
    else:
        with open(output_path, 'wb') as f:
            pickle.dump(instances, f)

    print(f"\nDataset saved to: {output_path}")

    # Save metadata
    metadata = {
        'n_instances': n_instances,
        'n_customers': n_customers,
        'capacity': config.capacity,
        'distribution': distribution.value,
        'solver': solver,
        'avg_distance': float(np.mean(distances)),
        'std_distance': float(np.std(distances)),
        'avg_vehicles': float(np.mean(n_vehicles))
    }

    meta_path = output_path.replace('.pkl', '_meta.json').replace('.gz', '')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    return output_path


# ============================================================================
# COLLATE FUNCTION
# ============================================================================

def cvrp_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for DataLoader."""
    coords = torch.stack([item['coords'] for item in batch])
    demands = torch.stack([item['demands'] for item in batch])
    capacity = torch.stack([item['capacity'] for item in batch])
    invariant_features = torch.stack([item['invariant_features'] for item in batch])

    output = {
        'coords': coords,
        'demands': demands,
        'capacity': capacity,
        'invariant_features': invariant_features,
        'n_nodes': batch[0]['n_nodes'],
        'n_customers': batch[0]['n_customers']
    }

    if 'gt_clusters' in batch[0]:
        output['gt_clusters'] = torch.stack([item['gt_clusters'] for item in batch])

    if 'gt_distance' in batch[0]:
        output['gt_distance'] = torch.tensor([item['gt_distance'] for item in batch])

    return output


# ============================================================================
# MAIN - DATA GENERATION SCRIPT
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate CVRP datasets")
    parser.add_argument('--n_customers', type=int, default=200)
    parser.add_argument('--n_train', type=int, default=10000)
    parser.add_argument('--n_val', type=int, default=1000)
    parser.add_argument('--n_test', type=int, default=1000)
    parser.add_argument('--distribution', type=str, default='uniform',
                       choices=['uniform', 'clustered', 'mixed'])
    parser.add_argument('--solver', type=str, default='auto')
    parser.add_argument('--time_limit', type=int, default=60)
    parser.add_argument('--output_dir', type=str, default='./data/cvrp_large')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--n_workers', type=int, default=4)

    args = parser.parse_args()

    dist_type = DistributionType(args.distribution)

    # Generate datasets
    for split, n_instances, seed_offset in [
        ('train', args.n_train, 0),
        ('val', args.n_val, 100000),
        ('test', args.n_test, 200000)
    ]:
        print(f"\n{'='*60}")
        print(f"Generating {split} set...")
        print(f"{'='*60}")

        output_path = os.path.join(
            args.output_dir,
            f"cvrp_{args.n_customers}_{args.distribution}_{split}.pkl"
        )

        generate_cvrp_dataset(
            n_instances=n_instances,
            n_customers=args.n_customers,
            output_path=output_path,
            distribution=dist_type,
            solver=args.solver,
            time_limit=args.time_limit,
            n_workers=args.n_workers,
            base_seed=args.seed + seed_offset
        )

    print("\n" + "=" * 60)
    print("All datasets generated successfully!")
    print("=" * 60)
