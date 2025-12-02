"""
EDISCO Routing Integration for Partition Framework
==================================================

This module integrates the pretrained EDISCO diffusion model with the
equivariant partition network for end-to-end large-scale CVRP solving.

Components:
1. EDISCORouter: Wrapper for pretrained EDISCO model
2. ClusterRouter: Routes individual clusters using various methods
3. EDISCOPartitionSolver: Complete partition + routing framework
4. Solution refinement utilities

Author: EDISCO Team
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import time
import warnings
from pathlib import Path

# Import partition model
from .model import (
    EquivariantPartitionNet,
    PartitionConfig,
    PartitionLoss,
    compute_route_distance,
    nearest_neighbor_route
)


# ============================================================================
# CONFIGURATION
# ============================================================================

class RoutingMethod(Enum):
    """Methods for routing within clusters."""
    EDISCO = "edisco"
    NEAREST_NEIGHBOR = "nearest_neighbor"
    TWO_OPT = "two_opt"
    LKH = "lkh"


@dataclass
class SolverConfig:
    """Configuration for the complete solver."""
    # Partition config
    n_clusters: int = 10
    partition_n_layers: int = 8
    partition_hidden_dim: int = 256

    # Routing config
    routing_method: RoutingMethod = RoutingMethod.NEAREST_NEIGHBOR
    edisco_checkpoint: Optional[str] = None
    edisco_n_steps: int = 50

    # Refinement
    use_two_opt: bool = True
    two_opt_iterations: int = 100

    # Device
    device: str = 'cuda'


# ============================================================================
# EDISCO MODEL LOADER
# ============================================================================

class EDISCOModelLoader:
    """
    Load pretrained EDISCO model for routing.

    Supports loading from:
    - TSP checkpoints (for cluster routing)
    - CVRP checkpoints (if available)
    """

    @staticmethod
    def load_tsp_model(
        checkpoint_path: str,
        device: str = 'cuda'
    ) -> Optional[nn.Module]:
        """
        Load pretrained TSP EDISCO model.

        The model architecture must match the checkpoint.
        """
        if not Path(checkpoint_path).exists():
            warnings.warn(f"Checkpoint not found: {checkpoint_path}")
            return None

        try:
            # Import the model architecture
            # This assumes the model definition is available
            from edisco_tsp_model import ContinuousTimeDiffusionTSPSolver

            checkpoint = torch.load(checkpoint_path, map_location=device)

            # Infer model configuration from checkpoint
            state_dict = checkpoint if isinstance(checkpoint, dict) and 'state_dict' not in checkpoint else checkpoint.get('state_dict', checkpoint)

            # Try to infer hidden dim from first layer
            for key in state_dict.keys():
                if 'node_embed' in key and 'weight' in key:
                    hidden_dim = state_dict[key].shape[0]
                    break
            else:
                hidden_dim = 256

            # Create model
            model = ContinuousTimeDiffusionTSPSolver(
                n_layers=12,
                hidden_dim=hidden_dim
            )

            model.load_state_dict(state_dict)
            model = model.to(device)
            model.eval()

            return model

        except Exception as e:
            warnings.warn(f"Failed to load EDISCO model: {e}")
            return None

    @staticmethod
    def create_minimal_router(device: str = 'cuda') -> nn.Module:
        """
        Create a minimal diffusion-based router for clusters.

        This is a lightweight model that can be trained from scratch
        or used as fallback when pretrained model is unavailable.
        """
        from edisco_partition_model import EGNNLayer

        class MinimalDiffusionRouter(nn.Module):
            """Lightweight diffusion router for small clusters."""

            def __init__(self, hidden_dim: int = 64, n_layers: int = 4):
                super().__init__()
                self.hidden_dim = hidden_dim

                # Node embedding
                self.node_embed = nn.Sequential(
                    nn.Linear(2, hidden_dim),  # coords
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )

                # Edge embedding
                self.edge_embed = nn.Linear(1, hidden_dim)

                # EGNN layers
                self.layers = nn.ModuleList([
                    EGNNLayer(hidden_dim, hidden_dim, hidden_dim, coord_dim=2, dropout=0.0, use_attention=False)
                    for _ in range(n_layers)
                ])

                # Output
                self.output = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.SiLU(),
                    nn.Linear(hidden_dim // 2, 2)  # Binary edge prediction
                )

            def forward(self, coords: torch.Tensor) -> torch.Tensor:
                """
                Predict edge probabilities for TSP.

                Args:
                    coords: (batch, n_nodes, 2)

                Returns:
                    edge_probs: (batch, n_nodes, n_nodes)
                """
                batch_size, n_nodes, _ = coords.shape

                # Embed nodes (using coords as features - simple version)
                h = self.node_embed(coords)

                # Edge features from distances
                dist = torch.cdist(coords, coords)
                e = self.edge_embed(dist.unsqueeze(-1))

                # Keep coords for EGNN
                x = coords.clone()

                # Apply layers
                for layer in self.layers:
                    h, x, e = layer(h, x, e)

                # Predict edges from edge features
                edge_logits = self.output(e)  # (batch, n, n, 2)
                edge_probs = F.softmax(edge_logits, dim=-1)[..., 1]  # Probability of edge

                return edge_probs

        return MinimalDiffusionRouter().to(device)


# ============================================================================
# CLUSTER ROUTING
# ============================================================================

class ClusterRouter:
    """
    Routes customers within a cluster using various methods.
    """

    def __init__(
        self,
        method: RoutingMethod = RoutingMethod.NEAREST_NEIGHBOR,
        edisco_model: Optional[nn.Module] = None,
        edisco_n_steps: int = 50,
        device: str = 'cuda'
    ):
        self.method = method
        self.edisco_model = edisco_model
        self.edisco_n_steps = edisco_n_steps
        self.device = device

    def route(
        self,
        coords: np.ndarray,
        customers: List[int],
        demands: Optional[np.ndarray] = None,
        capacity: Optional[float] = None
    ) -> List[int]:
        """
        Route customers in a cluster.

        Args:
            coords: Full coordinate array (n_nodes, 2)
            customers: List of customer indices in this cluster
            demands: Optional demand array
            capacity: Optional capacity

        Returns:
            Ordered list of customer indices
        """
        if len(customers) <= 1:
            return customers

        if self.method == RoutingMethod.NEAREST_NEIGHBOR:
            return self._route_nearest_neighbor(coords, customers)
        elif self.method == RoutingMethod.TWO_OPT:
            route = self._route_nearest_neighbor(coords, customers)
            return self._improve_two_opt(coords, route)
        elif self.method == RoutingMethod.EDISCO and self.edisco_model is not None:
            return self._route_edisco(coords, customers)
        else:
            return self._route_nearest_neighbor(coords, customers)

    def _route_nearest_neighbor(
        self,
        coords: np.ndarray,
        customers: List[int]
    ) -> List[int]:
        """Nearest neighbor heuristic."""
        route = []
        remaining = set(customers)
        current_pos = coords[0]  # Start at depot

        while remaining:
            nearest = min(remaining, key=lambda c: np.linalg.norm(current_pos - coords[c]))
            route.append(nearest)
            current_pos = coords[nearest]
            remaining.remove(nearest)

        return route

    def _improve_two_opt(
        self,
        coords: np.ndarray,
        route: List[int],
        max_iterations: int = 100
    ) -> List[int]:
        """2-opt improvement."""
        if len(route) <= 3:
            return route

        def route_distance(r):
            total = np.linalg.norm(coords[0] - coords[r[0]])
            for i in range(len(r) - 1):
                total += np.linalg.norm(coords[r[i]] - coords[r[i+1]])
            total += np.linalg.norm(coords[r[-1]] - coords[0])
            return total

        best_route = route[:]
        best_dist = route_distance(best_route)
        improved = True
        iterations = 0

        while improved and iterations < max_iterations:
            improved = False
            iterations += 1

            for i in range(len(best_route) - 1):
                for j in range(i + 2, len(best_route)):
                    new_route = best_route[:i+1] + best_route[i+1:j+1][::-1] + best_route[j+1:]
                    new_dist = route_distance(new_route)

                    if new_dist < best_dist - 1e-10:
                        best_route = new_route
                        best_dist = new_dist
                        improved = True

        return best_route

    def _route_edisco(
        self,
        coords: np.ndarray,
        customers: List[int]
    ) -> List[int]:
        """Route using EDISCO diffusion model."""
        if self.edisco_model is None:
            return self._route_nearest_neighbor(coords, customers)

        # Extract sub-problem coordinates
        indices = [0] + customers  # Include depot
        sub_coords = coords[indices]

        # Convert to tensor
        sub_coords_tensor = torch.FloatTensor(sub_coords).unsqueeze(0).to(self.device)

        # Get edge probabilities
        with torch.no_grad():
            edge_probs = self.edisco_model(sub_coords_tensor)[0]

        # Greedy decoding
        n = len(indices)
        edge_probs_np = edge_probs.cpu().numpy()

        visited = [False] * n
        visited[0] = True  # Depot
        route = []
        current = 0

        for _ in range(n - 1):
            # Find best unvisited neighbor
            best_prob = -1
            best_next = -1

            for j in range(1, n):  # Skip depot
                if not visited[j]:
                    prob = edge_probs_np[current, j] + edge_probs_np[j, current]
                    if prob > best_prob:
                        best_prob = prob
                        best_next = j

            if best_next == -1:
                break

            visited[best_next] = True
            route.append(customers[best_next - 1])  # Map back to original index
            current = best_next

        # Add any missed customers
        for i, customer in enumerate(customers):
            if customer not in route:
                route.append(customer)

        return route


# ============================================================================
# COMPLETE SOLVER
# ============================================================================

class EDISCOPartitionSolver(nn.Module):
    """
    Complete EDISCO-Partition solver for large-scale CVRP.

    Pipeline:
    1. Partition: EquivariantPartitionNet clusters customers
    2. Route: ClusterRouter optimizes each cluster
    3. Assemble: Combine routes into full solution
    4. Refine: Optional inter-cluster optimization
    """

    def __init__(self, config: SolverConfig):
        super().__init__()
        self.config = config

        # Partition network
        partition_config = PartitionConfig(
            n_clusters=config.n_clusters,
            n_layers=config.partition_n_layers,
            hidden_dim=config.partition_hidden_dim
        )
        self.partition_net = EquivariantPartitionNet(partition_config)

        # Cluster router
        edisco_model = None
        if config.edisco_checkpoint:
            edisco_model = EDISCOModelLoader.load_tsp_model(
                config.edisco_checkpoint, config.device
            )

        self.router = ClusterRouter(
            method=config.routing_method,
            edisco_model=edisco_model,
            edisco_n_steps=config.edisco_n_steps,
            device=config.device
        )

        # Loss function
        self.loss_fn = PartitionLoss(partition_config)

    def forward(
        self,
        coords: torch.Tensor,
        demands: torch.Tensor,
        capacity: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through partition network."""
        return self.partition_net(coords, demands, capacity)

    def solve(
        self,
        coords: torch.Tensor,
        demands: torch.Tensor,
        capacity: torch.Tensor,
        return_clusters: bool = False
    ) -> Dict[str, Union[List[List[List[int]]], torch.Tensor, float]]:
        """
        Solve CVRP instances.

        Args:
            coords: (batch, n_nodes, 2)
            demands: (batch, n_nodes)
            capacity: (batch,)
            return_clusters: Whether to return cluster assignments

        Returns:
            Dictionary with:
                - routes: List of routes per batch
                - distances: Total distances
                - clusters: (optional) Cluster assignments
        """
        batch_size = coords.shape[0]
        self.eval()

        with torch.no_grad():
            # Get partition
            outputs = self.partition_net(coords, demands, capacity)

            # Get hard cluster assignments
            clusters = self.partition_net.get_clusters(
                outputs['cluster_logits'],
                demands,
                capacity[0].item(),
                strategy='greedy'
            )

        # Route each cluster
        all_routes = []
        all_distances = []

        for b in range(batch_size):
            coords_np = coords[b].cpu().numpy()
            demands_np = demands[b].cpu().numpy()
            cap = capacity[b].item()

            batch_routes = []
            for cluster in clusters[b]:
                if not cluster:
                    continue

                # Check if cluster needs splitting
                cluster_demand = sum(demands_np[c] for c in cluster)

                if cluster_demand <= cap:
                    # Route the cluster
                    route = self.router.route(coords_np, cluster)
                    batch_routes.append(route)
                else:
                    # Split cluster and route each part
                    sub_routes = self._split_and_route(
                        coords_np, cluster, demands_np, cap
                    )
                    batch_routes.extend(sub_routes)

            # Optional 2-opt refinement
            if self.config.use_two_opt:
                batch_routes = [
                    self.router._improve_two_opt(coords_np, r, self.config.two_opt_iterations)
                    for r in batch_routes if r
                ]

            all_routes.append(batch_routes)
            all_distances.append(compute_route_distance(coords_np, batch_routes))

        result = {
            'routes': all_routes,
            'distances': all_distances
        }

        if return_clusters:
            result['clusters'] = clusters

        return result

    def _split_and_route(
        self,
        coords: np.ndarray,
        customers: List[int],
        demands: np.ndarray,
        capacity: float
    ) -> List[List[int]]:
        """Split over-capacity cluster and route each part."""
        # Sort by angle from depot for sweep-like splitting
        depot = coords[0]
        angles = [np.arctan2(coords[c, 1] - depot[1], coords[c, 0] - depot[0])
                  for c in customers]
        sorted_customers = [c for _, c in sorted(zip(angles, customers))]

        routes = []
        current_route = []
        current_load = 0

        for customer in sorted_customers:
            if current_load + demands[customer] <= capacity:
                current_route.append(customer)
                current_load += demands[customer]
            else:
                if current_route:
                    # Route and add
                    route = self.router.route(coords, current_route)
                    routes.append(route)
                current_route = [customer]
                current_load = demands[customer]

        if current_route:
            route = self.router.route(coords, current_route)
            routes.append(route)

        return routes

    def compute_loss(
        self,
        coords: torch.Tensor,
        demands: torch.Tensor,
        capacity: torch.Tensor,
        gt_clusters: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute training loss."""
        outputs = self.forward(coords, demands, capacity)
        return self.loss_fn(outputs, coords, demands, capacity, gt_clusters)


# ============================================================================
# BASELINE METHODS
# ============================================================================

class KMeansPartitionSolver:
    """
    K-means based partition solver (baseline for comparison).
    """

    def __init__(self, n_clusters: int = 10, device: str = 'cuda'):
        self.n_clusters = n_clusters
        self.device = device

    def solve(
        self,
        coords: torch.Tensor,
        demands: torch.Tensor,
        capacity: torch.Tensor
    ) -> Dict[str, Union[List, float]]:
        """Solve using K-means clustering."""
        batch_size = coords.shape[0]

        all_routes = []
        all_distances = []

        for b in range(batch_size):
            coords_np = coords[b].cpu().numpy()
            demands_np = demands[b].cpu().numpy()
            cap = capacity[b].item()

            # K-means clustering
            clusters = self._kmeans_cluster(coords_np[1:], self.n_clusters)

            # Adjust indices (add 1 for depot offset)
            clusters = [[c + 1 for c in cluster] for cluster in clusters]

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
                    # Split
                    sub_routes = self._split_cluster(coords_np, cluster, demands_np, cap)
                    routes.extend(sub_routes)

            all_routes.append(routes)
            all_distances.append(compute_route_distance(coords_np, routes))

        return {
            'routes': all_routes,
            'distances': all_distances
        }

    def _kmeans_cluster(
        self,
        coords: np.ndarray,
        k: int,
        max_iter: int = 100
    ) -> List[List[int]]:
        """Simple K-means clustering."""
        n = len(coords)
        k = min(k, n)

        # Initialize centroids randomly
        indices = np.random.choice(n, k, replace=False)
        centroids = coords[indices].copy()

        for _ in range(max_iter):
            # Assign points to nearest centroid
            distances = np.linalg.norm(coords[:, None] - centroids[None], axis=2)
            assignments = distances.argmin(axis=1)

            # Update centroids
            new_centroids = np.zeros_like(centroids)
            counts = np.zeros(k)

            for i, cluster_id in enumerate(assignments):
                new_centroids[cluster_id] += coords[i]
                counts[cluster_id] += 1

            for j in range(k):
                if counts[j] > 0:
                    new_centroids[j] /= counts[j]
                else:
                    new_centroids[j] = centroids[j]

            if np.allclose(centroids, new_centroids):
                break

            centroids = new_centroids

        # Create cluster lists
        clusters = [[] for _ in range(k)]
        for i, cluster_id in enumerate(assignments):
            clusters[cluster_id].append(i)

        return [c for c in clusters if c]

    def _split_cluster(
        self,
        coords: np.ndarray,
        customers: List[int],
        demands: np.ndarray,
        capacity: float
    ) -> List[List[int]]:
        """Split cluster by capacity."""
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


class SweepPartitionSolver:
    """
    Sweep algorithm based solver (baseline).
    """

    def solve(
        self,
        coords: torch.Tensor,
        demands: torch.Tensor,
        capacity: torch.Tensor
    ) -> Dict[str, Union[List, float]]:
        """Solve using sweep algorithm."""
        batch_size = coords.shape[0]

        all_routes = []
        all_distances = []

        for b in range(batch_size):
            coords_np = coords[b].cpu().numpy()
            demands_np = demands[b].cpu().numpy()
            cap = capacity[b].item()

            depot = coords_np[0]
            n_customers = len(coords_np) - 1

            # Compute angles from depot
            angles = []
            for i in range(1, n_customers + 1):
                dx = coords_np[i, 0] - depot[0]
                dy = coords_np[i, 1] - depot[1]
                angles.append((np.arctan2(dy, dx), i))

            angles.sort()

            # Create routes by sweeping
            routes = []
            current_route = []
            current_load = 0

            for _, customer in angles:
                if current_load + demands_np[customer] <= cap:
                    current_route.append(customer)
                    current_load += demands_np[customer]
                else:
                    if current_route:
                        routes.append(current_route)
                    current_route = [customer]
                    current_load = demands_np[customer]

            if current_route:
                routes.append(current_route)

            all_routes.append(routes)
            all_distances.append(compute_route_distance(coords_np, routes))

        return {
            'routes': all_routes,
            'distances': all_distances
        }


# ============================================================================
# SOLUTION COMPARISON
# ============================================================================

def compare_solvers(
    coords: torch.Tensor,
    demands: torch.Tensor,
    capacity: torch.Tensor,
    edisco_solver: EDISCOPartitionSolver,
    gt_distance: Optional[float] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compare different solvers on the same instances.

    Returns dictionary of solver_name -> metrics
    """
    results = {}

    # EDISCO-Partition
    start = time.time()
    edisco_result = edisco_solver.solve(coords, demands, capacity)
    edisco_time = time.time() - start

    results['edisco_partition'] = {
        'distance': np.mean(edisco_result['distances']),
        'time': edisco_time,
        'n_routes': np.mean([len(r) for r in edisco_result['routes']])
    }

    # K-means baseline
    kmeans_solver = KMeansPartitionSolver(n_clusters=edisco_solver.config.n_clusters)
    start = time.time()
    kmeans_result = kmeans_solver.solve(coords, demands, capacity)
    kmeans_time = time.time() - start

    results['kmeans_partition'] = {
        'distance': np.mean(kmeans_result['distances']),
        'time': kmeans_time,
        'n_routes': np.mean([len(r) for r in kmeans_result['routes']])
    }

    # Sweep baseline
    sweep_solver = SweepPartitionSolver()
    start = time.time()
    sweep_result = sweep_solver.solve(coords, demands, capacity)
    sweep_time = time.time() - start

    results['sweep'] = {
        'distance': np.mean(sweep_result['distances']),
        'time': sweep_time,
        'n_routes': np.mean([len(r) for r in sweep_result['routes']])
    }

    # Compute gaps if ground truth available
    if gt_distance is not None:
        for name, metrics in results.items():
            metrics['gap'] = (metrics['distance'] - gt_distance) / gt_distance * 100

    return results


# ============================================================================
# MAIN TEST
# ============================================================================

if __name__ == "__main__":
    print("Testing EDISCO-Partition Solver...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Create solver
    config = SolverConfig(
        n_clusters=10,
        partition_n_layers=6,
        partition_hidden_dim=128,
        routing_method=RoutingMethod.TWO_OPT,
        device=device
    )

    solver = EDISCOPartitionSolver(config).to(device)

    n_params = sum(p.numel() for p in solver.parameters())
    print(f"Parameters: {n_params:,}")

    # Test on random instance
    batch_size = 2
    n_nodes = 201  # 200 customers + depot

    coords = torch.rand(batch_size, n_nodes, 2, device=device)
    coords[:, 0, :] = 0.5  # Depot at center

    demands = torch.randint(1, 10, (batch_size, n_nodes), device=device).float()
    demands[:, 0] = 0

    capacity = torch.full((batch_size,), 80.0, device=device)

    print(f"\nTest instance: {n_nodes - 1} customers")
    print(f"Total demand: {demands[0, 1:].sum().item():.0f}")
    print(f"Capacity: {capacity[0].item()}")
    print(f"Expected routes: ~{demands[0, 1:].sum().item() / capacity[0].item():.1f}")

    # Solve
    print("\nSolving...")
    result = solver.solve(coords, demands, capacity, return_clusters=True)

    print(f"\nResults (batch 0):")
    print(f"  Routes: {len(result['routes'][0])}")
    print(f"  Distance: {result['distances'][0]:.3f}")
    print(f"  Clusters: {len(result['clusters'][0])}")

    # Compare with baselines
    print("\nComparing with baselines...")
    comparison = compare_solvers(coords, demands, capacity, solver)

    for name, metrics in comparison.items():
        print(f"\n{name}:")
        print(f"  Distance: {metrics['distance']:.3f}")
        print(f"  Time: {metrics['time']:.3f}s")
        print(f"  Routes: {metrics['n_routes']:.1f}")

    print("\nTest complete!")
