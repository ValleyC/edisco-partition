"""
Evaluation Framework for EDISCO-Partition
==========================================

Comprehensive evaluation with:
1. Multiple baseline comparisons (K-Means, Sweep, GLOP-style polar)
2. Statistical analysis (mean, std, confidence intervals)
3. Scalability analysis across problem sizes
4. Publication-ready tables and figures
5. Detailed per-instance analysis

Usage:
    python evaluate_partition.py --checkpoint checkpoints/partition/best_model.pt --n_instances 100
    python evaluate_partition.py --all_sizes --output_dir results/

Author: EDISCO Team
"""

import os
import sys
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict, field
import warnings
import pickle
import gzip

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy import stats

# Package imports
from edisco_partition import (
    EquivariantPartitionNet,
    PartitionConfig,
    verify_partition_equivariance,
    compute_route_distance,
    nearest_neighbor_route,
    EDISCOPartitionSolver,
    SolverConfig,
    RoutingMethod,
    KMeansPartitionSolver,
    SweepPartitionSolver,
    ClusterRouter,
    LargeCVRPDataset,
    cvrp_collate_fn,
    generate_cvrp_dataset,
    DistributionType
)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""
    # Problem settings
    problem_sizes: List[int] = field(default_factory=lambda: [200, 500, 1000])
    distributions: List[str] = field(default_factory=lambda: ['uniform', 'clustered', 'mixed'])
    n_instances: int = 100

    # Model checkpoint
    checkpoint_path: str = ""

    # Evaluation settings
    batch_size: int = 8
    use_two_opt: bool = True
    two_opt_iterations: int = 100

    # Output
    output_dir: str = "./results/evaluation"
    save_per_instance: bool = True
    generate_tables: bool = True
    generate_figures: bool = True

    # Hardware
    device: str = "cuda"
    seed: int = 42


# ============================================================================
# ADDITIONAL BASELINES
# ============================================================================

class PolarPartitionSolver:
    """
    GLOP-style polar coordinate partition (non-equivariant baseline).

    This uses polar coordinates (r, θ) from depot which is NOT equivariant
    under rotation - exactly what GLOP does.
    """

    def __init__(self, n_clusters: int = 10, use_angle_sectors: bool = True):
        self.n_clusters = n_clusters
        self.use_angle_sectors = use_angle_sectors

    def solve(
        self,
        coords: torch.Tensor,
        demands: torch.Tensor,
        capacity: torch.Tensor
    ) -> Dict[str, Any]:
        """Solve using polar coordinate partitioning."""
        batch_size = coords.shape[0]

        all_routes = []
        all_distances = []

        for b in range(batch_size):
            coords_np = coords[b].cpu().numpy()
            demands_np = demands[b].cpu().numpy()
            cap = capacity[b].item()

            depot = coords_np[0]
            n_customers = len(coords_np) - 1

            # Convert to polar coordinates
            polar = []
            for i in range(1, n_customers + 1):
                dx = coords_np[i, 0] - depot[0]
                dy = coords_np[i, 1] - depot[1]
                r = np.sqrt(dx**2 + dy**2)
                theta = np.arctan2(dy, dx)
                polar.append((r, theta, i))

            if self.use_angle_sectors:
                # Divide by angle sectors (like GLOP)
                clusters = self._angle_sector_partition(polar, self.n_clusters)
            else:
                # Cluster by both r and theta
                clusters = self._polar_kmeans(polar, self.n_clusters)

            # Route each cluster with capacity checking
            routes = []
            for cluster in clusters:
                if not cluster:
                    continue

                cluster_demand = sum(demands_np[c] for c in cluster)

                if cluster_demand <= cap:
                    route = nearest_neighbor_route(coords_np, cluster)
                    routes.append(route)
                else:
                    sub_routes = self._split_cluster(coords_np, cluster, demands_np, cap)
                    routes.extend(sub_routes)

            all_routes.append(routes)
            all_distances.append(compute_route_distance(coords_np, routes))

        return {
            'routes': all_routes,
            'distances': all_distances
        }

    def _angle_sector_partition(
        self,
        polar: List[Tuple[float, float, int]],
        k: int
    ) -> List[List[int]]:
        """Partition by dividing angle space into sectors."""
        # Sort by angle
        polar.sort(key=lambda x: x[1])

        n = len(polar)
        customers_per_cluster = n // k

        clusters = []
        for i in range(k):
            start = i * customers_per_cluster
            if i == k - 1:
                end = n
            else:
                end = (i + 1) * customers_per_cluster

            cluster = [p[2] for p in polar[start:end]]
            if cluster:
                clusters.append(cluster)

        return clusters

    def _polar_kmeans(
        self,
        polar: List[Tuple[float, float, int]],
        k: int,
        max_iter: int = 50
    ) -> List[List[int]]:
        """K-means in polar coordinate space."""
        n = len(polar)
        k = min(k, n)

        # Normalize r to similar scale as theta
        r_values = [p[0] for p in polar]
        r_max = max(r_values) if r_values else 1.0

        # Convert to 2D polar features
        features = np.array([
            [p[0] / r_max, p[1] / np.pi]  # Normalize both to ~[-1, 1]
            for p in polar
        ])

        # K-means
        indices = np.random.choice(n, k, replace=False)
        centroids = features[indices].copy()

        for _ in range(max_iter):
            distances = np.linalg.norm(features[:, None] - centroids[None], axis=2)
            assignments = distances.argmin(axis=1)

            new_centroids = np.zeros_like(centroids)
            counts = np.zeros(k)

            for i, cluster_id in enumerate(assignments):
                new_centroids[cluster_id] += features[i]
                counts[cluster_id] += 1

            for j in range(k):
                if counts[j] > 0:
                    new_centroids[j] /= counts[j]
                else:
                    new_centroids[j] = centroids[j]

            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids

        clusters = [[] for _ in range(k)]
        for i, cluster_id in enumerate(assignments):
            clusters[cluster_id].append(polar[i][2])

        return [c for c in clusters if c]

    def _split_cluster(
        self,
        coords: np.ndarray,
        customers: List[int],
        demands: np.ndarray,
        capacity: float
    ) -> List[List[int]]:
        """Split over-capacity cluster."""
        routes = []
        current = []
        current_load = 0

        for c in customers:
            if current_load + demands[c] <= capacity:
                current.append(c)
                current_load += demands[c]
            else:
                if current:
                    routes.append(nearest_neighbor_route(coords, current))
                current = [c]
                current_load = demands[c]

        if current:
            routes.append(nearest_neighbor_route(coords, current))

        return routes


class RandomPartitionSolver:
    """Random partition baseline (lower bound reference)."""

    def __init__(self, n_clusters: int = 10, seed: int = 42):
        self.n_clusters = n_clusters
        self.rng = np.random.RandomState(seed)

    def solve(
        self,
        coords: torch.Tensor,
        demands: torch.Tensor,
        capacity: torch.Tensor
    ) -> Dict[str, Any]:
        """Solve using random clustering."""
        batch_size = coords.shape[0]

        all_routes = []
        all_distances = []

        for b in range(batch_size):
            coords_np = coords[b].cpu().numpy()
            demands_np = demands[b].cpu().numpy()
            cap = capacity[b].item()
            n_customers = len(coords_np) - 1

            # Random assignment
            assignments = self.rng.randint(0, self.n_clusters, n_customers)

            clusters = [[] for _ in range(self.n_clusters)]
            for i, cluster_id in enumerate(assignments):
                clusters[cluster_id].append(i + 1)  # +1 for depot offset

            # Route each cluster
            routes = []
            for cluster in clusters:
                if not cluster:
                    continue

                cluster_demand = sum(demands_np[c] for c in cluster)

                if cluster_demand <= cap:
                    route = nearest_neighbor_route(coords_np, cluster)
                    routes.append(route)
                else:
                    current = []
                    current_load = 0
                    for c in cluster:
                        if current_load + demands_np[c] <= cap:
                            current.append(c)
                            current_load += demands_np[c]
                        else:
                            if current:
                                routes.append(nearest_neighbor_route(coords_np, current))
                            current = [c]
                            current_load = demands_np[c]
                    if current:
                        routes.append(nearest_neighbor_route(coords_np, current))

            all_routes.append(routes)
            all_distances.append(compute_route_distance(coords_np, routes))

        return {
            'routes': all_routes,
            'distances': all_distances
        }


# ============================================================================
# EVALUATOR
# ============================================================================

class PartitionEvaluator:
    """
    Comprehensive evaluator for partition methods.
    """

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')

        # Set seed
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Results storage
        self.results = {}

    def load_model(self, checkpoint_path: str, n_clusters: int = 10) -> EDISCOPartitionSolver:
        """Load trained model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Get config from checkpoint
        ckpt_config = checkpoint.get('config', {})

        solver_config = SolverConfig(
            n_clusters=ckpt_config.get('n_clusters', n_clusters),
            partition_n_layers=ckpt_config.get('n_layers', 8),
            partition_hidden_dim=ckpt_config.get('hidden_dim', 256),
            routing_method=RoutingMethod.TWO_OPT if self.config.use_two_opt else RoutingMethod.NEAREST_NEIGHBOR,
            two_opt_iterations=self.config.two_opt_iterations,
            device=str(self.device)
        )

        solver = EDISCOPartitionSolver(solver_config).to(self.device)
        solver.partition_net.load_state_dict(checkpoint['model_state_dict'])
        solver.eval()

        return solver

    def create_baselines(self, n_clusters: int = 10) -> Dict[str, Any]:
        """Create baseline solvers."""
        return {
            'kmeans': KMeansPartitionSolver(n_clusters=n_clusters),
            'sweep': SweepPartitionSolver(),
            'polar_sector': PolarPartitionSolver(n_clusters=n_clusters, use_angle_sectors=True),
            'polar_kmeans': PolarPartitionSolver(n_clusters=n_clusters, use_angle_sectors=False),
            'random': RandomPartitionSolver(n_clusters=n_clusters, seed=self.config.seed)
        }

    def evaluate_solver(
        self,
        solver: Any,
        coords: torch.Tensor,
        demands: torch.Tensor,
        capacity: torch.Tensor,
        gt_distances: Optional[torch.Tensor] = None,
        solver_name: str = "solver"
    ) -> Dict[str, Any]:
        """Evaluate a single solver."""
        batch_size = coords.shape[0]

        # Warm-up (especially for CUDA)
        if batch_size >= 2:
            _ = solver.solve(coords[:1], demands[:1], capacity[:1])

        # Time the solve
        start_time = time.time()
        result = solver.solve(coords, demands, capacity)
        solve_time = time.time() - start_time

        distances = result['distances']
        routes = result['routes']

        # Compute metrics
        metrics = {
            'solver': solver_name,
            'batch_size': batch_size,
            'solve_time': solve_time,
            'time_per_instance': solve_time / batch_size,
            'distance_mean': np.mean(distances),
            'distance_std': np.std(distances),
            'distance_min': np.min(distances),
            'distance_max': np.max(distances),
            'n_routes_mean': np.mean([len(r) for r in routes]),
            'n_routes_std': np.std([len(r) for r in routes])
        }

        # Gap to ground truth if available
        if gt_distances is not None:
            gt_np = gt_distances.cpu().numpy()
            gaps = [(d - g) / g * 100 for d, g in zip(distances, gt_np)]
            metrics['gap_mean'] = np.mean(gaps)
            metrics['gap_std'] = np.std(gaps)
            metrics['gap_min'] = np.min(gaps)
            metrics['gap_max'] = np.max(gaps)

            # 95% confidence interval for gap
            if len(gaps) > 1:
                ci = stats.t.interval(0.95, len(gaps)-1,
                                      loc=np.mean(gaps),
                                      scale=stats.sem(gaps))
                metrics['gap_ci_lower'] = ci[0]
                metrics['gap_ci_upper'] = ci[1]

        # Per-instance results
        metrics['distances'] = distances
        metrics['routes'] = routes

        return metrics

    def evaluate_problem_size(
        self,
        n_customers: int,
        distribution: str,
        trained_solver: Optional[EDISCOPartitionSolver] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Evaluate all solvers on a specific problem configuration."""
        print(f"\n{'='*60}")
        print(f"Evaluating CVRP-{n_customers} ({distribution} distribution)")
        print(f"{'='*60}")

        # Determine number of clusters
        n_clusters = max(5, n_customers // 20)  # Roughly 20 customers per cluster

        # Load or generate test data
        data_dir = Path("./data/cvrp_large")
        test_path = data_dir / f"cvrp_{n_customers}_{distribution}_test.pkl.gz"

        if not test_path.exists():
            print(f"Generating test data at {test_path}...")
            generate_cvrp_dataset(
                n_instances=self.config.n_instances,
                n_customers=n_customers,
                output_path=str(test_path).replace('.gz', ''),
                distribution=DistributionType(distribution),
                solver='auto',
                time_limit=300,  # More time for large instances
                base_seed=self.config.seed + 200000
            )

        # Load dataset
        dataset = LargeCVRPDataset(
            str(test_path),
            augment=False,
            return_gt_clusters=True
        )

        # Limit to n_instances
        if len(dataset) > self.config.n_instances:
            indices = list(range(self.config.n_instances))
            dataset.data = [dataset.data[i] for i in indices]

        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=cvrp_collate_fn,
            num_workers=0  # Single-threaded for timing accuracy
        )

        # Create baselines
        baselines = self.create_baselines(n_clusters)

        # Add trained solver if available
        solvers = {}
        if trained_solver is not None:
            solvers['edisco_partition'] = trained_solver
        solvers.update(baselines)

        # Evaluate each solver
        results = {}

        for solver_name, solver in solvers.items():
            print(f"\nEvaluating {solver_name}...")

            all_metrics = []

            for batch in tqdm(dataloader, desc=solver_name):
                coords = batch['coords'].to(self.device)
                demands = batch['demands'].to(self.device)
                capacity = batch['capacity'].to(self.device)
                gt_distances = batch.get('gt_distance')

                metrics = self.evaluate_solver(
                    solver, coords, demands, capacity,
                    gt_distances, solver_name
                )
                all_metrics.append(metrics)

            # Aggregate metrics
            aggregated = self._aggregate_metrics(all_metrics)
            results[solver_name] = aggregated

            # Print summary
            print(f"  Distance: {aggregated['distance_mean']:.3f} ± {aggregated['distance_std']:.3f}")
            print(f"  Time/instance: {aggregated['time_per_instance']*1000:.1f}ms")
            if 'gap_mean' in aggregated:
                print(f"  Gap: {aggregated['gap_mean']:.2f}% ± {aggregated['gap_std']:.2f}%")

        return results

    def _aggregate_metrics(self, metrics_list: List[Dict]) -> Dict[str, Any]:
        """Aggregate metrics from multiple batches."""
        all_distances = []
        all_times = []
        all_gaps = []
        all_n_routes = []

        for m in metrics_list:
            all_distances.extend(m['distances'])
            all_times.append(m['solve_time'])
            if 'gap_mean' in m:
                # Recompute from raw if available
                pass
            all_n_routes.extend([len(r) for r in m['routes']])

        n_instances = len(all_distances)
        total_time = sum(all_times)

        result = {
            'n_instances': n_instances,
            'distance_mean': np.mean(all_distances),
            'distance_std': np.std(all_distances),
            'distance_min': np.min(all_distances),
            'distance_max': np.max(all_distances),
            'total_time': total_time,
            'time_per_instance': total_time / n_instances,
            'n_routes_mean': np.mean(all_n_routes),
            'n_routes_std': np.std(all_n_routes)
        }

        # Compute gaps if available
        if 'distances' in metrics_list[0] and any('gap_mean' in m for m in metrics_list):
            # Need original gt_distances to compute properly
            pass

        return result

    def run_full_evaluation(self) -> Dict[str, Any]:
        """Run complete evaluation across all configurations."""
        print("\n" + "="*70)
        print("EDISCO-Partition Full Evaluation")
        print("="*70)
        print(f"Output directory: {self.output_dir}")
        print(f"Problem sizes: {self.config.problem_sizes}")
        print(f"Distributions: {self.config.distributions}")
        print(f"Instances per config: {self.config.n_instances}")

        # Load trained model if checkpoint provided
        trained_solver = None
        if self.config.checkpoint_path and Path(self.config.checkpoint_path).exists():
            print(f"\nLoading model from {self.config.checkpoint_path}")
            trained_solver = self.load_model(self.config.checkpoint_path)

            # Verify equivariance
            print("Verifying equivariance...")
            eq_results = verify_partition_equivariance(
                trained_solver.partition_net,
                batch_size=2,
                n_nodes=201,
                device=str(self.device)
            )
            for test, passed in eq_results.items():
                status = "✓" if passed else "✗"
                print(f"  {test}: {status}")

        # Run evaluations
        all_results = {}

        for n_customers in self.config.problem_sizes:
            for distribution in self.config.distributions:
                key = f"cvrp{n_customers}_{distribution}"
                results = self.evaluate_problem_size(
                    n_customers, distribution, trained_solver
                )
                all_results[key] = results

                # Save intermediate results
                self._save_results(all_results)

        # Generate tables and figures
        if self.config.generate_tables:
            self._generate_latex_tables(all_results)

        if self.config.generate_figures:
            self._generate_figures(all_results)

        return all_results

    def _save_results(self, results: Dict):
        """Save results to file."""
        # Convert to serializable format
        serializable = {}
        for config_key, config_results in results.items():
            serializable[config_key] = {}
            for solver_name, metrics in config_results.items():
                serializable[config_key][solver_name] = {
                    k: v for k, v in metrics.items()
                    if k not in ['distances', 'routes']  # Skip large arrays
                }

        # Save as JSON
        json_path = self.output_dir / "results.json"
        with open(json_path, 'w') as f:
            json.dump(serializable, f, indent=2)

        # Save complete results as pickle
        pkl_path = self.output_dir / "results_complete.pkl"
        with open(pkl_path, 'wb') as f:
            pickle.dump(results, f)

    def _generate_latex_tables(self, results: Dict):
        """Generate LaTeX tables for publication."""
        print("\n" + "-"*60)
        print("Generating LaTeX tables...")

        # Main results table
        table_lines = [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Comparison of partition methods on CVRP benchmarks. " +
            r"Distance is total route length; Gap (\%) is relative to LKH solution.}",
            r"\label{tab:cvrp_results}",
            r"\small",
            r"\begin{tabular}{l" + "c"*len(self.config.problem_sizes)*2 + "}",
            r"\toprule"
        ]

        # Header
        header = "Method"
        for n in self.config.problem_sizes:
            header += f" & \\multicolumn{{2}}{{c}}{{CVRP-{n}}}"
        header += r" \\"
        table_lines.append(header)

        subheader = ""
        for _ in self.config.problem_sizes:
            subheader += " & Distance & Time"
        subheader += r" \\"
        table_lines.append(subheader)
        table_lines.append(r"\midrule")

        # Get solver names
        first_key = list(results.keys())[0]
        solver_names = list(results[first_key].keys())

        # Nice names for solvers
        nice_names = {
            'edisco_partition': r'\textbf{EDISCO-Partition (Ours)}',
            'kmeans': 'K-Means',
            'sweep': 'Sweep',
            'polar_sector': 'Polar Sector (GLOP-style)',
            'polar_kmeans': 'Polar K-Means',
            'random': 'Random'
        }

        for solver_name in solver_names:
            row = nice_names.get(solver_name, solver_name)

            for n in self.config.problem_sizes:
                key = f"cvrp{n}_uniform"  # Use uniform for main table
                if key in results and solver_name in results[key]:
                    metrics = results[key][solver_name]
                    dist = metrics.get('distance_mean', 0)
                    time_ms = metrics.get('time_per_instance', 0) * 1000
                    row += f" & {dist:.2f} & {time_ms:.1f}ms"
                else:
                    row += " & - & -"

            row += r" \\"
            table_lines.append(row)

        table_lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}"
        ])

        # Save table
        table_path = self.output_dir / "results_table.tex"
        with open(table_path, 'w') as f:
            f.write('\n'.join(table_lines))

        print(f"Saved table to {table_path}")

        # Also print to console
        print("\n" + "-"*40)
        print("LaTeX Table:")
        print("-"*40)
        print('\n'.join(table_lines))

    def _generate_figures(self, results: Dict):
        """Generate figures for publication."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')
        except ImportError:
            print("matplotlib not available, skipping figures")
            return

        print("\n" + "-"*60)
        print("Generating figures...")

        # Figure 1: Distance comparison bar chart
        fig, ax = plt.subplots(figsize=(10, 6))

        solver_names = []
        for key in results:
            solver_names = list(results[key].keys())
            break

        x = np.arange(len(self.config.problem_sizes))
        width = 0.15

        colors = {
            'edisco_partition': '#2ecc71',
            'kmeans': '#3498db',
            'sweep': '#9b59b6',
            'polar_sector': '#e74c3c',
            'polar_kmeans': '#f39c12',
            'random': '#95a5a6'
        }

        for i, solver_name in enumerate(solver_names):
            distances = []
            for n in self.config.problem_sizes:
                key = f"cvrp{n}_uniform"
                if key in results and solver_name in results[key]:
                    distances.append(results[key][solver_name].get('distance_mean', 0))
                else:
                    distances.append(0)

            offset = (i - len(solver_names)/2 + 0.5) * width
            bars = ax.bar(x + offset, distances, width,
                         label=solver_name.replace('_', ' ').title(),
                         color=colors.get(solver_name, '#7f8c8d'))

        ax.set_xlabel('Problem Size', fontsize=12)
        ax.set_ylabel('Total Route Distance', fontsize=12)
        ax.set_title('Partition Method Comparison on CVRP', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels([f'CVRP-{n}' for n in self.config.problem_sizes])
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        fig_path = self.output_dir / "distance_comparison.pdf"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved figure to {fig_path}")

        # Figure 2: Scalability (time vs problem size)
        fig, ax = plt.subplots(figsize=(8, 5))

        for solver_name in solver_names:
            times = []
            sizes = []
            for n in self.config.problem_sizes:
                key = f"cvrp{n}_uniform"
                if key in results and solver_name in results[key]:
                    times.append(results[key][solver_name].get('time_per_instance', 0) * 1000)
                    sizes.append(n)

            if times:
                ax.plot(sizes, times, 'o-',
                       label=solver_name.replace('_', ' ').title(),
                       color=colors.get(solver_name, '#7f8c8d'),
                       linewidth=2, markersize=8)

        ax.set_xlabel('Number of Customers', fontsize=12)
        ax.set_ylabel('Time per Instance (ms)', fontsize=12)
        ax.set_title('Scalability Analysis', fontsize=14)
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_yscale('log')

        plt.tight_layout()
        fig_path = self.output_dir / "scalability.pdf"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved figure to {fig_path}")


# ============================================================================
# EQUIVARIANCE COMPARISON
# ============================================================================

def compare_equivariance(
    edisco_solver: EDISCOPartitionSolver,
    polar_solver: PolarPartitionSolver,
    n_rotations: int = 8,
    n_instances: int = 10,
    n_customers: int = 100,
    device: str = 'cuda'
) -> Dict[str, Any]:
    """
    Compare equivariance properties of different partition methods.

    Shows that EDISCO-Partition produces rotation-invariant cluster assignments
    while polar coordinate methods do not.
    """
    print("\n" + "="*60)
    print("Equivariance Comparison")
    print("="*60)

    device = torch.device(device)

    results = {
        'edisco': {'variance': [], 'distance_variance': []},
        'polar': {'variance': [], 'distance_variance': []}
    }

    for inst in range(n_instances):
        # Generate random instance
        coords = torch.rand(1, n_customers + 1, 2, device=device)
        coords[:, 0] = 0.5  # Depot at center
        demands = torch.randint(1, 10, (1, n_customers + 1), device=device).float()
        demands[:, 0] = 0
        capacity = torch.tensor([50.0], device=device)

        edisco_distances = []
        polar_distances = []

        for r in range(n_rotations):
            angle = 2 * np.pi * r / n_rotations

            # Rotate coordinates around depot
            rotated = coords.clone()
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rotation = torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]],
                                   dtype=torch.float32, device=device)

            depot = coords[:, 0:1, :]
            relative = coords - depot
            rotated_relative = torch.matmul(relative, rotation.T)
            rotated = depot + rotated_relative

            # EDISCO
            edisco_result = edisco_solver.solve(rotated, demands, capacity)
            edisco_distances.append(edisco_result['distances'][0])

            # Polar
            polar_result = polar_solver.solve(rotated, demands, capacity)
            polar_distances.append(polar_result['distances'][0])

        # Compute variance across rotations (lower = more equivariant)
        results['edisco']['distance_variance'].append(np.var(edisco_distances))
        results['polar']['distance_variance'].append(np.var(polar_distances))

    # Aggregate
    edisco_var = np.mean(results['edisco']['distance_variance'])
    polar_var = np.mean(results['polar']['distance_variance'])

    print(f"\nDistance variance across rotations:")
    print(f"  EDISCO-Partition: {edisco_var:.6f}")
    print(f"  Polar (GLOP-style): {polar_var:.6f}")
    print(f"  Ratio: {polar_var/edisco_var:.2f}x higher variance for polar")

    return {
        'edisco_variance': edisco_var,
        'polar_variance': polar_var,
        'ratio': polar_var / edisco_var if edisco_var > 0 else float('inf')
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate EDISCO-Partition")

    # Input
    parser.add_argument('--checkpoint', type=str, default='',
                       help='Path to trained model checkpoint')

    # Problem settings
    parser.add_argument('--problem_sizes', type=int, nargs='+', default=[200],
                       help='Problem sizes to evaluate')
    parser.add_argument('--distributions', type=str, nargs='+', default=['uniform'],
                       help='Distribution types to evaluate')
    parser.add_argument('--n_instances', type=int, default=100,
                       help='Number of instances per configuration')
    parser.add_argument('--all_sizes', action='store_true',
                       help='Evaluate all standard sizes (200, 500, 1000)')

    # Output
    parser.add_argument('--output_dir', type=str, default='./results/evaluation',
                       help='Output directory for results')

    # Other
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--equivariance_test', action='store_true',
                       help='Run equivariance comparison')

    args = parser.parse_args()

    # Configure
    if args.all_sizes:
        args.problem_sizes = [200, 500, 1000]
        args.distributions = ['uniform', 'clustered', 'mixed']

    config = EvaluationConfig(
        problem_sizes=args.problem_sizes,
        distributions=args.distributions,
        n_instances=args.n_instances,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed,
        batch_size=args.batch_size
    )

    # Run evaluation
    evaluator = PartitionEvaluator(config)
    results = evaluator.run_full_evaluation()

    # Run equivariance comparison if requested
    if args.equivariance_test and config.checkpoint_path:
        trained_solver = evaluator.load_model(config.checkpoint_path)
        polar_solver = PolarPartitionSolver(n_clusters=10, use_angle_sectors=True)

        eq_results = compare_equivariance(
            trained_solver, polar_solver,
            n_rotations=8, n_instances=20, n_customers=100,
            device=args.device
        )

        # Save equivariance results
        eq_path = evaluator.output_dir / "equivariance_comparison.json"
        with open(eq_path, 'w') as f:
            json.dump(eq_results, f, indent=2)

    print("\n" + "="*60)
    print("Evaluation complete!")
    print(f"Results saved to {config.output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
