"""
EDISCO-Partition: E(2)-Equivariant Partition Network for Large-Scale CVRP
=========================================================================

This package implements E(2)-equivariant graph neural networks for partitioning
large-scale Capacitated Vehicle Routing Problem (CVRP) instances.

Key Features:
- Provable E(2)-equivariance (rotation, translation, reflection invariant)
- Scalable to CVRP-200, CVRP-500, CVRP-1000
- Multiple routing methods (Nearest Neighbor, 2-opt, EDISCO)
- Comprehensive baseline comparisons

Example:
    >>> from edisco_partition import EDISCOInferencePipeline, RandomCVRPGenerator
    >>>
    >>> # Generate random instance
    >>> instance = RandomCVRPGenerator.generate(n_customers=200)
    >>>
    >>> # Create pipeline and solve
    >>> pipeline = EDISCOInferencePipeline(checkpoint_path='best_model.pt')
    >>> solution = pipeline.solve(instance)
    >>> print(f"Distance: {solution['distance']:.4f}")
"""

__version__ = "0.1.0"
__author__ = "EDISCO Team"

# Core model components
from .model import (
    EquivariantPartitionNet,
    PartitionConfig,
    PartitionLoss,
    EGNNLayer,
    MultiScaleEGNN,
    CapacityAwareClusterHead,
    verify_partition_equivariance,
    compute_route_distance,
    nearest_neighbor_route,
    create_partition_model
)

# Solver components
from .solver import (
    EDISCOPartitionSolver,
    SolverConfig,
    RoutingMethod,
    ClusterRouter,
    KMeansPartitionSolver,
    SweepPartitionSolver,
    compare_solvers
)

# Data utilities
from .data import (
    LargeCVRPDataset,
    CVRPConfig,
    DistributionType,
    cvrp_collate_fn,
    generate_cvrp_dataset
)

# Inference pipeline
from .inference import (
    EDISCOInferencePipeline,
    CVRPInstanceReader,
    RandomCVRPGenerator,
    SolutionWriter,
    SolutionVisualizer
)

__all__ = [
    # Version info
    "__version__",
    "__author__",

    # Model
    "EquivariantPartitionNet",
    "PartitionConfig",
    "PartitionLoss",
    "EGNNLayer",
    "MultiScaleEGNN",
    "CapacityAwareClusterHead",
    "verify_partition_equivariance",
    "compute_route_distance",
    "nearest_neighbor_route",
    "create_partition_model",

    # Solver
    "EDISCOPartitionSolver",
    "SolverConfig",
    "RoutingMethod",
    "ClusterRouter",
    "KMeansPartitionSolver",
    "SweepPartitionSolver",
    "compare_solvers",

    # Data
    "LargeCVRPDataset",
    "CVRPConfig",
    "DistributionType",
    "cvrp_collate_fn",
    "generate_cvrp_dataset",

    # Inference
    "EDISCOInferencePipeline",
    "CVRPInstanceReader",
    "RandomCVRPGenerator",
    "SolutionWriter",
    "SolutionVisualizer",
]
