#!/usr/bin/env python
"""
Evaluation script for trained EDISCO-Partition models.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import numpy as np
from tqdm import tqdm

from edisco_partition.models.partition_net import PartitionNet
from edisco_partition.data.instance import generate_instance
from edisco_partition.data.graph import build_graph, build_graph_cosine
from edisco_partition.sampler.sequential import Sampler
from edisco_partition.evaluation.nn_routing import eval_routes_nn
from edisco_partition.evaluation.lkh_routing import eval_routes_lkh, LKH_PATH
from edisco_partition.utils.helpers import set_seed, get_device, check_equivariance


K_SPARSE = {
    100: 20, 200: 30, 500: 50, 1000: 100, 2000: 200,
}


def get_k_sparse(n):
    return K_SPARSE.get(n, min(200, max(20, n // 10)))


@torch.no_grad()
def evaluate(model, n_customers, n_instances, width, device, opts):
    """
    Evaluate model on random instances.

    Args:
        model: Trained PartitionNet
        n_customers: Problem size
        n_instances: Number of instances to evaluate
        width: Number of samples per instance
        device: Device
        opts: Options

    Returns:
        results: Dictionary with statistics
    """
    model.eval()

    costs = []
    k_sparse = get_k_sparse(n_customers)

    for _ in tqdm(range(n_instances), desc="Evaluating"):
        coords, demand, capacity = generate_instance(n_customers, device)

        if opts.use_cosine:
            graph = build_graph_cosine(coords, demand, capacity, k_sparse)
        else:
            graph = build_graph(coords, demand, capacity, k_sparse)

        heatmap = model(graph)

        if opts.greedy:
            # Single greedy sample
            sampler = Sampler(demand, heatmap, capacity, 1, device)
            routes = sampler.gen_subsets(require_prob=False, greedy_mode=True)
        else:
            # Multiple samples
            sampler = Sampler(demand, heatmap, capacity, width, device)
            routes = sampler.gen_subsets(require_prob=False, greedy_mode=False)

        # Evaluate
        if opts.use_lkh and LKH_PATH is not None:
            instance_costs = eval_routes_lkh(
                coords, routes, capacity, demand,
                time_limit=opts.lkh_time_limit,
                max_trials=opts.lkh_max_trials
            )
        else:
            instance_costs = eval_routes_nn(coords, routes)

        costs.append(instance_costs.min().item())

    costs = np.array(costs)

    return {
        'mean': costs.mean(),
        'std': costs.std(),
        'min': costs.min(),
        'max': costs.max(),
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate EDISCO-Partition')

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--problem_size', type=int, default=200,
                        help='Number of customers')
    parser.add_argument('--n_instances', type=int, default=100,
                        help='Number of instances to evaluate')
    parser.add_argument('--width', type=int, default=10,
                        help='Samples per instance')
    parser.add_argument('--greedy', action='store_true',
                        help='Use greedy decoding')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device')

    # Model settings (must match training)
    parser.add_argument('--depth', type=int, default=12)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--node_dim', type=int, default=64)
    parser.add_argument('--edge_dim', type=int, default=64)

    # Graph settings
    parser.add_argument('--use_cosine', action='store_true')

    # Evaluation settings
    parser.add_argument('--use_lkh', action='store_true')
    parser.add_argument('--lkh_time_limit', type=int, default=1)
    parser.add_argument('--lkh_max_trials', type=int, default=100)

    opts = parser.parse_args()

    device = get_device(opts.device)
    set_seed(opts.seed)

    # Load model
    model = PartitionNet(
        n_layers=opts.depth,
        hidden_dim=opts.hidden_dim,
        node_dim=opts.node_dim,
        edge_dim=opts.edge_dim,
    ).to(device)

    checkpoint = torch.load(opts.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")

    # Check equivariance
    coords, demand, capacity = generate_instance(opts.problem_size, device)
    equiv_diff = check_equivariance(
        model, coords, demand, capacity,
        get_k_sparse(opts.problem_size), device
    )
    print(f"Equivariance check: {equiv_diff:.6f}")

    # Evaluate
    print(f"\nEvaluating on {opts.n_instances} instances...")
    results = evaluate(model, opts.problem_size, opts.n_instances, opts.width, device, opts)

    print("\nResults:")
    print(f"  Mean cost: {results['mean']:.4f}")
    print(f"  Std: {results['std']:.4f}")
    print(f"  Min: {results['min']:.4f}")
    print(f"  Max: {results['max']:.4f}")


if __name__ == '__main__':
    main()
