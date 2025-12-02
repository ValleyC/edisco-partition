"""
Quick Start Example for EDISCO-Partition
========================================

This example demonstrates how to use the EDISCO-Partition package
to solve a random CVRP instance.

Usage:
    python examples/quick_start.py
"""

import torch
from edisco_partition import (
    EDISCOInferencePipeline,
    RandomCVRPGenerator,
    verify_partition_equivariance,
    EquivariantPartitionNet,
    PartitionConfig
)


def main():
    print("=" * 60)
    print("EDISCO-Partition Quick Start")
    print("=" * 60)

    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # 1. Generate a random CVRP instance
    print("\n1. Generating random CVRP instance...")
    instance = RandomCVRPGenerator.generate(
        n_customers=100,
        distribution='uniform',
        seed=42
    )
    print(f"   Customers: {len(instance['coords']) - 1}")
    print(f"   Total demand: {instance['demands'].sum():.0f}")
    print(f"   Capacity: {instance['capacity']}")

    # 2. Verify model equivariance
    print("\n2. Creating and verifying model equivariance...")
    config = PartitionConfig(n_clusters=5, n_layers=4, hidden_dim=64)
    model = EquivariantPartitionNet(config)

    eq_results = verify_partition_equivariance(model, batch_size=2, n_nodes=51, device='cpu')
    for test_name, passed in eq_results.items():
        status = "PASS" if passed else "FAIL"
        print(f"   {test_name}: {status}")

    # 3. Create inference pipeline and solve
    print("\n3. Solving with EDISCO-Partition...")
    pipeline = EDISCOInferencePipeline(
        n_clusters=5,
        n_layers=4,
        hidden_dim=64,
        routing_method='two_opt',
        device=device
    )

    solution = pipeline.solve(instance, verbose=False)

    print(f"\n4. Solution:")
    print(f"   Total distance: {solution['distance']:.4f}")
    print(f"   Number of routes: {solution['n_routes']}")
    print(f"   Solve time: {solution['solve_time']:.3f}s")

    # 5. Compare with baselines
    print("\n5. Comparing with baselines...")
    from edisco_partition import KMeansPartitionSolver, SweepPartitionSolver

    coords = torch.FloatTensor(instance['coords']).unsqueeze(0)
    demands = torch.FloatTensor(instance['demands']).unsqueeze(0)
    capacity = torch.FloatTensor([instance['capacity']])

    # K-Means baseline
    kmeans = KMeansPartitionSolver(n_clusters=5)
    kmeans_result = kmeans.solve(coords, demands, capacity)
    print(f"   K-Means: distance={kmeans_result['distances'][0]:.4f}")

    # Sweep baseline
    sweep = SweepPartitionSolver()
    sweep_result = sweep.solve(coords, demands, capacity)
    print(f"   Sweep: distance={sweep_result['distances'][0]:.4f}")

    print("\n" + "=" * 60)
    print("Quick start complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
