"""
End-to-End Inference Pipeline for EDISCO-Partition
===================================================

Complete inference pipeline for solving large-scale CVRP instances.

Features:
1. Multiple input formats (pickle, JSON, TSPLIB, random generation)
2. Configurable routing methods
3. Solution visualization
4. Output in multiple formats
5. Batch processing support
6. Interactive and CLI modes

Usage:
    # Solve from file
    python inference_pipeline.py --input data/cvrp_instance.pkl --checkpoint best_model.pt

    # Generate and solve random instance
    python inference_pipeline.py --random --n_customers 500 --checkpoint best_model.pt

    # Batch processing
    python inference_pipeline.py --input_dir data/instances/ --output_dir solutions/

Author: EDISCO Team
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import pickle
import gzip
import warnings

import numpy as np
import torch
import torch.nn as nn

# Local imports
from .model import (
    EquivariantPartitionNet,
    PartitionConfig,
    compute_route_distance,
    nearest_neighbor_route
)
from .solver import (
    EDISCOPartitionSolver,
    SolverConfig,
    RoutingMethod,
    ClusterRouter
)


# ============================================================================
# INPUT READERS
# ============================================================================

class CVRPInstanceReader:
    """
    Read CVRP instances from various formats.
    """

    @staticmethod
    def read(path: str) -> Dict[str, np.ndarray]:
        """
        Read CVRP instance from file.

        Returns:
            Dictionary with keys:
                - coords: (n_nodes, 2) depot at index 0
                - demands: (n_nodes,) depot demand is 0
                - capacity: scalar
        """
        path = Path(path)

        if path.suffix in ['.pkl', '.pickle']:
            return CVRPInstanceReader._read_pickle(path)
        elif path.suffix == '.gz':
            return CVRPInstanceReader._read_gzip_pickle(path)
        elif path.suffix == '.json':
            return CVRPInstanceReader._read_json(path)
        elif path.suffix == '.vrp':
            return CVRPInstanceReader._read_tsplib(path)
        else:
            raise ValueError(f"Unknown file format: {path.suffix}")

    @staticmethod
    def _read_pickle(path: Path) -> Dict[str, np.ndarray]:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return CVRPInstanceReader._parse_data(data)

    @staticmethod
    def _read_gzip_pickle(path: Path) -> Dict[str, np.ndarray]:
        with gzip.open(path, 'rb') as f:
            data = pickle.load(f)
        return CVRPInstanceReader._parse_data(data)

    @staticmethod
    def _read_json(path: Path) -> Dict[str, np.ndarray]:
        with open(path, 'r') as f:
            data = json.load(f)

        coords = np.array(data['coords'], dtype=np.float32)
        demands = np.array(data['demands'], dtype=np.float32)
        capacity = float(data['capacity'])

        return {
            'coords': coords,
            'demands': demands,
            'capacity': capacity
        }

    @staticmethod
    def _read_tsplib(path: Path) -> Dict[str, np.ndarray]:
        """Read TSPLIB format CVRP instance."""
        coords = []
        demands = []
        capacity = 0
        n_nodes = 0
        depot_idx = 0

        with open(path, 'r') as f:
            section = None
            for line in f:
                line = line.strip()

                if line.startswith('CAPACITY'):
                    capacity = int(line.split(':')[1].strip())
                elif line.startswith('DIMENSION'):
                    n_nodes = int(line.split(':')[1].strip())
                elif line == 'NODE_COORD_SECTION':
                    section = 'coords'
                elif line == 'DEMAND_SECTION':
                    section = 'demands'
                elif line == 'DEPOT_SECTION':
                    section = 'depot'
                elif line == 'EOF':
                    break
                elif section == 'coords' and line:
                    parts = line.split()
                    if len(parts) >= 3:
                        coords.append([float(parts[1]), float(parts[2])])
                elif section == 'demands' and line:
                    parts = line.split()
                    if len(parts) >= 2:
                        demands.append(float(parts[1]))
                elif section == 'depot' and line:
                    try:
                        val = int(line)
                        if val > 0:
                            depot_idx = val - 1
                    except ValueError:
                        pass

        coords = np.array(coords, dtype=np.float32)
        demands = np.array(demands, dtype=np.float32)

        # Move depot to index 0 if needed
        if depot_idx != 0:
            coords[[0, depot_idx]] = coords[[depot_idx, 0]]
            demands[[0, depot_idx]] = demands[[depot_idx, 0]]

        # Normalize coordinates to [0, 1]
        coord_min = coords.min(axis=0)
        coord_max = coords.max(axis=0)
        coord_range = coord_max - coord_min
        coord_range[coord_range == 0] = 1
        coords = (coords - coord_min) / coord_range

        return {
            'coords': coords,
            'demands': demands,
            'capacity': capacity
        }

    @staticmethod
    def _parse_data(data: Any) -> Dict[str, np.ndarray]:
        """Parse data from loaded pickle."""
        if isinstance(data, dict):
            if 'coords' in data:
                return {
                    'coords': np.array(data['coords'], dtype=np.float32),
                    'demands': np.array(data['demands'], dtype=np.float32),
                    'capacity': float(data['capacity'])
                }
            elif 'depot' in data:
                # Handle format from some generators
                depot = np.array(data['depot'], dtype=np.float32).reshape(1, 2)
                customers = np.array(data['loc'], dtype=np.float32)
                coords = np.vstack([depot, customers])

                depot_demand = np.array([0], dtype=np.float32)
                customer_demands = np.array(data['demand'], dtype=np.float32)
                demands = np.concatenate([depot_demand, customer_demands])

                return {
                    'coords': coords,
                    'demands': demands,
                    'capacity': float(data['capacity'])
                }

        # Try as list of instances - return first one
        if isinstance(data, list) and len(data) > 0:
            return CVRPInstanceReader._parse_data(data[0])

        raise ValueError("Could not parse CVRP data format")


class RandomCVRPGenerator:
    """Generate random CVRP instances."""

    @staticmethod
    def generate(
        n_customers: int = 200,
        distribution: str = 'uniform',
        capacity: Optional[int] = None,
        seed: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Generate random CVRP instance.

        Args:
            n_customers: Number of customers
            distribution: uniform, clustered, mixed
            capacity: Vehicle capacity (auto-set if None)
            seed: Random seed

        Returns:
            Dictionary with coords, demands, capacity
        """
        if seed is not None:
            np.random.seed(seed)

        # Set capacity based on problem size
        if capacity is None:
            if n_customers <= 100:
                capacity = 40
            elif n_customers <= 200:
                capacity = 50
            elif n_customers <= 500:
                capacity = 80
            else:
                capacity = 100

        # Generate coordinates
        if distribution == 'uniform':
            coords = np.random.rand(n_customers + 1, 2).astype(np.float32)
        elif distribution == 'clustered':
            n_clusters = max(3, n_customers // 30)
            centers = np.random.rand(n_clusters, 2)
            assignments = np.random.randint(0, n_clusters, n_customers)
            coords = np.zeros((n_customers + 1, 2), dtype=np.float32)
            coords[0] = [0.5, 0.5]  # Depot at center
            for i in range(n_customers):
                center = centers[assignments[i]]
                offset = np.random.randn(2) * 0.1
                coords[i + 1] = np.clip(center + offset, 0, 1)
        elif distribution == 'mixed':
            # Half clustered, half uniform
            coords = np.random.rand(n_customers + 1, 2).astype(np.float32)
            n_clustered = n_customers // 2
            n_clusters = max(2, n_clustered // 20)
            centers = np.random.rand(n_clusters, 2)
            for i in range(n_clustered):
                center = centers[np.random.randint(n_clusters)]
                offset = np.random.randn(2) * 0.08
                coords[i + 1] = np.clip(center + offset, 0, 1)
        else:
            coords = np.random.rand(n_customers + 1, 2).astype(np.float32)

        # Depot at center
        coords[0] = [0.5, 0.5]

        # Generate demands
        demands = np.zeros(n_customers + 1, dtype=np.float32)
        demands[1:] = np.random.randint(1, 10, n_customers).astype(np.float32)

        return {
            'coords': coords,
            'demands': demands,
            'capacity': float(capacity)
        }


# ============================================================================
# SOLUTION OUTPUT
# ============================================================================

class SolutionWriter:
    """Write CVRP solutions to various formats."""

    @staticmethod
    def write(
        solution: Dict[str, Any],
        path: str,
        format: str = 'auto'
    ):
        """
        Write solution to file.

        Args:
            solution: Dictionary with routes, distance, etc.
            path: Output file path
            format: json, txt, or auto (infer from extension)
        """
        path = Path(path)

        if format == 'auto':
            format = path.suffix.lstrip('.')
            if not format:
                format = 'json'

        if format == 'json':
            SolutionWriter._write_json(solution, path)
        elif format == 'txt':
            SolutionWriter._write_text(solution, path)
        else:
            SolutionWriter._write_json(solution, path)

    @staticmethod
    def _write_json(solution: Dict, path: Path):
        # Convert numpy arrays to lists
        serializable = {}
        for k, v in solution.items():
            if isinstance(v, np.ndarray):
                serializable[k] = v.tolist()
            elif isinstance(v, (list, tuple)):
                serializable[k] = [
                    item.tolist() if isinstance(item, np.ndarray) else item
                    for item in v
                ]
            else:
                serializable[k] = v

        with open(path, 'w') as f:
            json.dump(serializable, f, indent=2)

    @staticmethod
    def _write_text(solution: Dict, path: Path):
        lines = []
        lines.append(f"CVRP Solution")
        lines.append(f"=" * 40)
        lines.append(f"Total Distance: {solution.get('distance', 'N/A'):.4f}")
        lines.append(f"Number of Routes: {len(solution.get('routes', []))}")
        lines.append(f"Solve Time: {solution.get('solve_time', 'N/A'):.3f}s")
        lines.append("")
        lines.append("Routes:")
        lines.append("-" * 40)

        for i, route in enumerate(solution.get('routes', [])):
            route_str = " -> ".join([str(c) for c in route])
            lines.append(f"Route {i+1}: 0 -> {route_str} -> 0")

        with open(path, 'w') as f:
            f.write('\n'.join(lines))


# ============================================================================
# VISUALIZATION
# ============================================================================

class SolutionVisualizer:
    """Visualize CVRP solutions."""

    @staticmethod
    def plot_solution(
        coords: np.ndarray,
        routes: List[List[int]],
        clusters: Optional[List[List[int]]] = None,
        title: str = "CVRP Solution",
        save_path: Optional[str] = None,
        show: bool = True
    ):
        """
        Plot CVRP solution.

        Args:
            coords: (n_nodes, 2)
            routes: List of routes
            clusters: Optional cluster assignments
            title: Plot title
            save_path: Path to save figure
            show: Whether to display plot
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.colors as mcolors
        except ImportError:
            print("matplotlib not available, skipping visualization")
            return

        fig, axes = plt.subplots(1, 2 if clusters else 1,
                                figsize=(12 if clusters else 8, 6))

        if not clusters:
            axes = [axes]

        # Color palette
        colors = list(mcolors.TABLEAU_COLORS.values())

        # Plot 1: Routes
        ax = axes[0]
        ax.scatter(coords[0, 0], coords[0, 1], c='red', s=200,
                  marker='s', label='Depot', zorder=10)
        ax.scatter(coords[1:, 0], coords[1:, 1], c='gray', s=30,
                  alpha=0.5, label='Customers')

        for i, route in enumerate(routes):
            color = colors[i % len(colors)]
            full_route = [0] + route + [0]
            for j in range(len(full_route) - 1):
                start = coords[full_route[j]]
                end = coords[full_route[j + 1]]
                ax.annotate('', xy=end, xytext=start,
                           arrowprops=dict(arrowstyle='->', color=color,
                                          alpha=0.7, lw=1.5))

        ax.set_title(f'{title}\n{len(routes)} routes', fontsize=12)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_aspect('equal')
        ax.legend(loc='upper right')

        # Plot 2: Clusters (if provided)
        if clusters:
            ax = axes[1]
            ax.scatter(coords[0, 0], coords[0, 1], c='red', s=200,
                      marker='s', label='Depot', zorder=10)

            for i, cluster in enumerate(clusters):
                color = colors[i % len(colors)]
                cluster_coords = coords[cluster]
                ax.scatter(cluster_coords[:, 0], cluster_coords[:, 1],
                          c=color, s=40, label=f'Cluster {i+1}')

            ax.set_title(f'Partition\n{len(clusters)} clusters', fontsize=12)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_aspect('equal')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved figure to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()


# ============================================================================
# INFERENCE PIPELINE
# ============================================================================

class EDISCOInferencePipeline:
    """
    Complete inference pipeline for EDISCO-Partition.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        n_clusters: int = 10,
        n_layers: int = 8,
        hidden_dim: int = 256,
        routing_method: str = 'two_opt',
        two_opt_iterations: int = 100,
        device: str = 'cuda'
    ):
        """
        Initialize pipeline.

        Args:
            checkpoint_path: Path to trained model checkpoint
            n_clusters: Number of clusters (if no checkpoint)
            n_layers: Number of EGNN layers (if no checkpoint)
            hidden_dim: Hidden dimension (if no checkpoint)
            routing_method: nearest_neighbor, two_opt, or edisco
            two_opt_iterations: Number of 2-opt iterations
            device: Device to run on
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Parse routing method
        routing_map = {
            'nearest_neighbor': RoutingMethod.NEAREST_NEIGHBOR,
            'nn': RoutingMethod.NEAREST_NEIGHBOR,
            'two_opt': RoutingMethod.TWO_OPT,
            '2opt': RoutingMethod.TWO_OPT,
            'edisco': RoutingMethod.EDISCO
        }
        routing = routing_map.get(routing_method.lower(), RoutingMethod.TWO_OPT)

        # Load or create model
        if checkpoint_path and Path(checkpoint_path).exists():
            self.solver = self._load_from_checkpoint(checkpoint_path, routing, two_opt_iterations)
        else:
            solver_config = SolverConfig(
                n_clusters=n_clusters,
                partition_n_layers=n_layers,
                partition_hidden_dim=hidden_dim,
                routing_method=routing,
                two_opt_iterations=two_opt_iterations,
                device=str(self.device)
            )
            self.solver = EDISCOPartitionSolver(solver_config).to(self.device)

        self.solver.eval()

        print(f"Pipeline initialized on {self.device}")
        n_params = sum(p.numel() for p in self.solver.parameters())
        print(f"Model parameters: {n_params:,}")

    def _load_from_checkpoint(
        self,
        path: str,
        routing: RoutingMethod,
        two_opt_iterations: int
    ) -> EDISCOPartitionSolver:
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        # Get config
        ckpt_config = checkpoint.get('config', {})

        solver_config = SolverConfig(
            n_clusters=ckpt_config.get('n_clusters', 10),
            partition_n_layers=ckpt_config.get('n_layers', 8),
            partition_hidden_dim=ckpt_config.get('hidden_dim', 256),
            routing_method=routing,
            two_opt_iterations=two_opt_iterations,
            device=str(self.device)
        )

        solver = EDISCOPartitionSolver(solver_config).to(self.device)
        solver.partition_net.load_state_dict(checkpoint['model_state_dict'])

        print(f"Loaded model from {path}")
        return solver

    def solve(
        self,
        instance: Dict[str, np.ndarray],
        return_clusters: bool = True,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Solve a CVRP instance.

        Args:
            instance: Dictionary with coords, demands, capacity
            return_clusters: Whether to return cluster assignments
            verbose: Whether to print progress

        Returns:
            Dictionary with routes, distance, clusters, solve_time
        """
        # Convert to tensors
        coords = torch.FloatTensor(instance['coords']).unsqueeze(0).to(self.device)
        demands = torch.FloatTensor(instance['demands']).unsqueeze(0).to(self.device)
        capacity = torch.FloatTensor([instance['capacity']]).to(self.device)

        n_customers = coords.shape[1] - 1

        if verbose:
            print(f"\nSolving CVRP instance:")
            print(f"  Customers: {n_customers}")
            print(f"  Total demand: {demands[0, 1:].sum().item():.0f}")
            print(f"  Capacity: {capacity.item():.0f}")
            print(f"  Min routes needed: {int(np.ceil(demands[0, 1:].sum().item() / capacity.item()))}")

        # Solve
        start_time = time.time()
        result = self.solver.solve(coords, demands, capacity, return_clusters=return_clusters)
        solve_time = time.time() - start_time

        routes = result['routes'][0]
        distance = result['distances'][0]
        clusters = result.get('clusters', [[]])[0] if return_clusters else None

        if verbose:
            print(f"\nSolution:")
            print(f"  Routes: {len(routes)}")
            print(f"  Total distance: {distance:.4f}")
            print(f"  Solve time: {solve_time:.3f}s")

            if clusters:
                print(f"  Clusters: {len(clusters)}")
                cluster_sizes = [len(c) for c in clusters]
                print(f"  Cluster sizes: min={min(cluster_sizes)}, max={max(cluster_sizes)}, "
                      f"avg={np.mean(cluster_sizes):.1f}")

        return {
            'routes': routes,
            'distance': distance,
            'clusters': clusters,
            'solve_time': solve_time,
            'n_routes': len(routes),
            'n_customers': n_customers
        }

    def solve_batch(
        self,
        instances: List[Dict[str, np.ndarray]],
        return_clusters: bool = False,
        verbose: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Solve multiple CVRP instances.

        Args:
            instances: List of instances
            return_clusters: Whether to return cluster assignments
            verbose: Whether to print progress

        Returns:
            List of solution dictionaries
        """
        solutions = []

        if verbose:
            print(f"\nSolving {len(instances)} instances...")

        for i, instance in enumerate(instances):
            if verbose:
                print(f"\nInstance {i+1}/{len(instances)}")

            solution = self.solve(instance, return_clusters, verbose=verbose)
            solutions.append(solution)

        if verbose:
            avg_dist = np.mean([s['distance'] for s in solutions])
            avg_time = np.mean([s['solve_time'] for s in solutions])
            print(f"\n" + "="*40)
            print(f"Batch Summary:")
            print(f"  Average distance: {avg_dist:.4f}")
            print(f"  Average time: {avg_time:.3f}s")
            print(f"  Total time: {sum(s['solve_time'] for s in solutions):.3f}s")

        return solutions

    def solve_from_file(
        self,
        path: str,
        return_clusters: bool = True,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Solve CVRP instance from file.

        Args:
            path: Path to input file
            return_clusters: Whether to return cluster assignments
            verbose: Whether to print progress

        Returns:
            Solution dictionary
        """
        if verbose:
            print(f"Reading instance from {path}...")

        instance = CVRPInstanceReader.read(path)
        return self.solve(instance, return_clusters, verbose)

    def solve_and_visualize(
        self,
        instance: Dict[str, np.ndarray],
        save_path: Optional[str] = None,
        show: bool = True
    ) -> Dict[str, Any]:
        """
        Solve and visualize solution.

        Args:
            instance: CVRP instance
            save_path: Path to save figure
            show: Whether to display

        Returns:
            Solution dictionary
        """
        solution = self.solve(instance, return_clusters=True, verbose=True)

        SolutionVisualizer.plot_solution(
            coords=instance['coords'],
            routes=solution['routes'],
            clusters=solution['clusters'],
            title=f"EDISCO-Partition Solution (dist={solution['distance']:.3f})",
            save_path=save_path,
            show=show
        )

        return solution


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="EDISCO-Partition Inference Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Solve from file
  python inference_pipeline.py --input instance.pkl --checkpoint best_model.pt

  # Generate and solve random instance
  python inference_pipeline.py --random --n_customers 500 --checkpoint best_model.pt

  # Batch process directory
  python inference_pipeline.py --input_dir instances/ --output_dir solutions/

  # Visualize solution
  python inference_pipeline.py --input instance.pkl --checkpoint best_model.pt --visualize
        """
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument('--input', type=str, help='Input instance file')
    input_group.add_argument('--input_dir', type=str, help='Directory of input instances')
    input_group.add_argument('--random', action='store_true', help='Generate random instance')

    # Random generation options
    parser.add_argument('--n_customers', type=int, default=200,
                       help='Number of customers for random generation')
    parser.add_argument('--distribution', type=str, default='uniform',
                       choices=['uniform', 'clustered', 'mixed'],
                       help='Distribution type for random generation')
    parser.add_argument('--capacity', type=int, default=None,
                       help='Vehicle capacity (auto-set if not specified)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed')

    # Model options
    parser.add_argument('--checkpoint', type=str, default='',
                       help='Path to model checkpoint')
    parser.add_argument('--n_clusters', type=int, default=10,
                       help='Number of clusters (if no checkpoint)')
    parser.add_argument('--n_layers', type=int, default=8,
                       help='Number of EGNN layers')
    parser.add_argument('--hidden_dim', type=int, default=256,
                       help='Hidden dimension')

    # Routing options
    parser.add_argument('--routing', type=str, default='two_opt',
                       choices=['nearest_neighbor', 'nn', 'two_opt', '2opt', 'edisco'],
                       help='Routing method')
    parser.add_argument('--two_opt_iterations', type=int, default=100,
                       help='Number of 2-opt iterations')

    # Output options
    parser.add_argument('--output', type=str, help='Output solution file')
    parser.add_argument('--output_dir', type=str, help='Output directory for batch')
    parser.add_argument('--output_format', type=str, default='json',
                       choices=['json', 'txt'],
                       help='Output format')

    # Visualization
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize solution')
    parser.add_argument('--save_figure', type=str,
                       help='Path to save visualization')
    parser.add_argument('--no_show', action='store_true',
                       help='Do not display visualization')

    # Other
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run on')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress verbose output')

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = EDISCOInferencePipeline(
        checkpoint_path=args.checkpoint,
        n_clusters=args.n_clusters,
        n_layers=args.n_layers,
        hidden_dim=args.hidden_dim,
        routing_method=args.routing,
        two_opt_iterations=args.two_opt_iterations,
        device=args.device
    )

    verbose = not args.quiet

    # Handle different input modes
    if args.random:
        # Generate random instance
        instance = RandomCVRPGenerator.generate(
            n_customers=args.n_customers,
            distribution=args.distribution,
            capacity=args.capacity,
            seed=args.seed
        )

        if args.visualize:
            solution = pipeline.solve_and_visualize(
                instance,
                save_path=args.save_figure,
                show=not args.no_show
            )
        else:
            solution = pipeline.solve(instance, verbose=verbose)

        if args.output:
            SolutionWriter.write(solution, args.output, args.output_format)
            print(f"Solution saved to {args.output}")

    elif args.input:
        # Solve from file
        instance = CVRPInstanceReader.read(args.input)

        if args.visualize:
            solution = pipeline.solve_and_visualize(
                instance,
                save_path=args.save_figure,
                show=not args.no_show
            )
        else:
            solution = pipeline.solve(instance, verbose=verbose)

        if args.output:
            SolutionWriter.write(solution, args.output, args.output_format)
            print(f"Solution saved to {args.output}")

    elif args.input_dir:
        # Batch process directory
        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir) if args.output_dir else input_dir / 'solutions'
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find all instance files
        extensions = ['.pkl', '.pickle', '.json', '.vrp', '.gz']
        files = []
        for ext in extensions:
            files.extend(input_dir.glob(f'*{ext}'))

        print(f"Found {len(files)} instance files")

        for file_path in files:
            print(f"\nProcessing {file_path.name}...")

            try:
                instance = CVRPInstanceReader.read(str(file_path))
                solution = pipeline.solve(instance, verbose=verbose)

                # Save solution
                output_path = output_dir / f"{file_path.stem}_solution.{args.output_format}"
                SolutionWriter.write(solution, str(output_path), args.output_format)

                # Save visualization if requested
                if args.visualize:
                    fig_path = output_dir / f"{file_path.stem}_solution.png"
                    SolutionVisualizer.plot_solution(
                        coords=instance['coords'],
                        routes=solution['routes'],
                        clusters=solution.get('clusters'),
                        title=f"{file_path.stem} (dist={solution['distance']:.3f})",
                        save_path=str(fig_path),
                        show=False
                    )

            except Exception as e:
                print(f"Error processing {file_path}: {e}")

        print(f"\nSolutions saved to {output_dir}")

    else:
        # Interactive demo
        print("\nNo input specified. Running demo with random instance...")

        instance = RandomCVRPGenerator.generate(
            n_customers=100,
            distribution='uniform',
            seed=42
        )

        solution = pipeline.solve_and_visualize(
            instance,
            save_path=args.save_figure,
            show=not args.no_show
        )


if __name__ == "__main__":
    main()
