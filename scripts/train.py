#!/usr/bin/env python
"""
Training script for E(2)-Equivariant CVRP Partition.

Uses REINFORCE (policy gradient) with the following pipeline:
1. Generate CVRP instance
2. Build E(2)-equivariant graph (no polar angle!)
3. Predict heatmap with EGNN
4. Sample routes using sequential sampler
5. Evaluate with NN routing (fast) or LKH (optional)
6. Update with REINFORCE: loss = (cost - baseline) * (-log_prob)
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from edisco_partition.models.partition_net import PartitionNet
from edisco_partition.data.instance import generate_instance, CAPACITIES
from edisco_partition.data.graph import build_graph, build_graph_cosine
from edisco_partition.sampler.sequential import Sampler
from edisco_partition.evaluation.nn_routing import eval_routes_nn
from edisco_partition.evaluation.lkh_routing import eval_routes_lkh, LKH_PATH
from edisco_partition.utils.helpers import (
    set_seed, get_device, count_parameters,
    save_checkpoint, load_checkpoint, AverageMeter, check_equivariance
)


# Default k_sparse values for different problem sizes
K_SPARSE = {
    100: 20,
    200: 30,
    500: 50,
    1000: 100,
    2000: 200,
    5000: 200,
}


def get_k_sparse(n):
    """Get k_sparse value for problem size n."""
    if n in K_SPARSE:
        return K_SPARSE[n]
    # Interpolate for unknown sizes
    return min(200, max(20, n // 10))


def train_step(model, optimizer, n_customers, batch_size, width, device, opts):
    """
    Single training step.

    Args:
        model: PartitionNet
        optimizer: Optimizer
        n_customers: Problem size
        batch_size: Number of instances per step
        width: Number of samples per instance
        device: Device
        opts: Training options

    Returns:
        loss: Training loss value
        avg_cost: Average route cost
    """
    model.train()

    loss_list = []
    cost_list = []

    for _ in range(batch_size):
        # Generate instance
        coords, demand, capacity = generate_instance(n_customers, device)

        # Build E(2)-equivariant graph
        k_sparse = get_k_sparse(n_customers)
        if opts.use_cosine:
            graph = build_graph_cosine(coords, demand, capacity, k_sparse)
        else:
            graph = build_graph(coords, demand, capacity, k_sparse)

        # Predict heatmap
        heatmap = model(graph)

        # Sample routes
        sampler = Sampler(demand, heatmap, capacity, width, device)
        routes, log_probs = sampler.gen_subsets(require_prob=True)

        # Evaluate routes
        if opts.use_lkh and LKH_PATH is not None:
            costs = eval_routes_lkh(
                coords, routes, capacity, demand,
                time_limit=opts.lkh_time_limit,
                max_trials=opts.lkh_max_trials
            )
        else:
            costs = eval_routes_nn(coords, routes)

        # REINFORCE
        baseline = costs.mean()
        log_probs_sum = log_probs.sum(dim=1)  # Sum over time steps
        advantage = costs - baseline

        # Loss: (cost - baseline) * (-log_prob) = -advantage * log_prob
        # We want to minimize cost, so good samples (low cost) should have
        # negative advantage and thus increase their log_prob
        reinforce_loss = (advantage * log_probs_sum).mean()

        loss_list.append(reinforce_loss)
        cost_list.append(costs.mean().item())

    # Average over batch
    loss = sum(loss_list) / batch_size

    # Backprop
    optimizer.zero_grad()
    loss.backward()

    if not opts.no_clip:
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=opts.max_norm)

    optimizer.step()

    return loss.item(), np.mean(cost_list)


@torch.no_grad()
def validate(model, n_customers, val_size, device, opts):
    """
    Validation loop.

    Args:
        model: PartitionNet
        n_customers: Problem size
        val_size: Number of validation instances
        device: Device
        opts: Options

    Returns:
        avg_cost: Average validation cost
    """
    model.eval()

    total_cost = 0.0
    k_sparse = get_k_sparse(n_customers)

    for _ in range(val_size):
        coords, demand, capacity = generate_instance(n_customers, device)

        if opts.use_cosine:
            graph = build_graph_cosine(coords, demand, capacity, k_sparse)
        else:
            graph = build_graph(coords, demand, capacity, k_sparse)

        heatmap = model(graph)

        # Greedy sampling for validation
        sampler = Sampler(demand, heatmap, capacity, 1, device)
        routes = sampler.gen_subsets(require_prob=False, greedy_mode=True)

        if opts.use_lkh and LKH_PATH is not None:
            cost = eval_routes_lkh(
                coords, routes, capacity, demand,
                time_limit=opts.lkh_time_limit,
                max_trials=opts.lkh_max_trials
            ).min()
        else:
            cost = eval_routes_nn(coords, routes).min()

        total_cost += cost.item()

    return total_cost / val_size


def train(opts):
    """Main training function."""
    device = get_device(opts.device)
    set_seed(opts.seed)

    # Create model
    model = PartitionNet(
        n_layers=opts.depth,
        hidden_dim=opts.hidden_dim,
        node_dim=opts.node_dim,
        edge_dim=opts.edge_dim,
        node_input_dim=2,  # (demand/capacity, distance_from_depot)
        edge_input_dim=2,  # (distance, affinity)
        coord_update_alpha=opts.coord_alpha,
        weight_temp=opts.weight_temp
    ).to(device)

    print(f"Model parameters: {count_parameters(model):,}")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=opts.n_epochs, eta_min=opts.lr * 0.01
    )

    # Load checkpoint if provided
    start_epoch = 1
    best_val_cost = float('inf')

    if opts.checkpoint:
        start_epoch = load_checkpoint(opts.checkpoint, model, optimizer, scheduler) + 1
        print(f"Resumed from epoch {start_epoch - 1}")

    # Create save directory
    save_dir = os.path.join('checkpoints', f'cvrp{opts.problem_size}')
    os.makedirs(save_dir, exist_ok=True)

    # Initial validation
    val_cost = validate(model, opts.problem_size, opts.val_size, device, opts)
    print(f"Initial validation cost: {val_cost:.4f}")
    best_val_cost = val_cost

    # Check equivariance
    coords, demand, capacity = generate_instance(opts.problem_size, device)
    equiv_diff = check_equivariance(
        model, coords, demand, capacity,
        get_k_sparse(opts.problem_size), device
    )
    print(f"Equivariance check (should be ~0): {equiv_diff:.6f}")

    # Training loop
    print("\nStarting training...")
    print(f"Problem size: {opts.problem_size}")
    print(f"Epochs: {opts.n_epochs}")
    print(f"Steps per epoch: {opts.steps_per_epoch}")
    print(f"Batch size: {opts.batch_size}")
    print(f"Width (samples): {opts.width}")
    print(f"Using LKH: {opts.use_lkh and LKH_PATH is not None}")
    print("-" * 60)

    for epoch in range(start_epoch, opts.n_epochs + 1):
        epoch_start = time.time()
        loss_meter = AverageMeter()
        cost_meter = AverageMeter()

        pbar = tqdm(range(opts.steps_per_epoch), desc=f"Epoch {epoch}")
        for step in pbar:
            loss, cost = train_step(
                model, optimizer,
                opts.problem_size, opts.batch_size, opts.width,
                device, opts
            )
            loss_meter.update(loss)
            cost_meter.update(cost)

            pbar.set_postfix({
                'loss': f'{loss_meter.avg:.4f}',
                'cost': f'{cost_meter.avg:.2f}'
            })

        scheduler.step()

        epoch_time = time.time() - epoch_start

        # Validation
        val_cost = validate(model, opts.problem_size, opts.val_size, device, opts)

        # Check equivariance periodically
        if epoch % 5 == 0:
            coords, demand, capacity = generate_instance(opts.problem_size, device)
            equiv_diff = check_equivariance(
                model, coords, demand, capacity,
                get_k_sparse(opts.problem_size), device
            )
        else:
            equiv_diff = -1

        print(f"Epoch {epoch}: loss={loss_meter.avg:.4f}, "
              f"train_cost={cost_meter.avg:.2f}, val_cost={val_cost:.2f}, "
              f"time={epoch_time:.1f}s" +
              (f", equiv={equiv_diff:.4f}" if equiv_diff >= 0 else ""))

        # Save best model
        if val_cost < best_val_cost:
            best_val_cost = val_cost
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                os.path.join(save_dir, 'best.pt'),
                val_cost=val_cost
            )
            print(f"  -> New best model saved (val_cost={val_cost:.4f})")

        # Save periodic checkpoint
        if epoch % opts.save_freq == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                os.path.join(save_dir, f'epoch_{epoch}.pt'),
                val_cost=val_cost
            )

    print("\nTraining complete!")
    print(f"Best validation cost: {best_val_cost:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Train E(2)-Equivariant CVRP Partition')

    # Problem settings
    parser.add_argument('--problem_size', type=int, default=200,
                        help='Number of customers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (auto/cuda/cpu)')

    # Training settings
    parser.add_argument('--n_epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--steps_per_epoch', type=int, default=256,
                        help='Steps per epoch')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='Instances per step')
    parser.add_argument('--width', type=int, default=10,
                        help='Samples per instance')
    parser.add_argument('--val_size', type=int, default=100,
                        help='Validation instances')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='Save checkpoint every N epochs')

    # Model settings
    parser.add_argument('--depth', type=int, default=12,
                        help='Number of EGNN layers')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension')
    parser.add_argument('--node_dim', type=int, default=64,
                        help='Node embedding dimension')
    parser.add_argument('--edge_dim', type=int, default=64,
                        help='Edge embedding dimension')
    parser.add_argument('--coord_alpha', type=float, default=0.1,
                        help='Coordinate update step size')
    parser.add_argument('--weight_temp', type=float, default=10.0,
                        help='Weight temperature for coordinate updates')

    # Optimization settings
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--max_norm', type=float, default=1.0,
                        help='Gradient clipping max norm')
    parser.add_argument('--no_clip', action='store_true',
                        help='Disable gradient clipping')

    # Graph settings
    parser.add_argument('--use_cosine', action='store_true',
                        help='Use cosine similarity for graph construction')

    # Evaluation settings
    parser.add_argument('--use_lkh', action='store_true',
                        help='Use LKH for evaluation (slower but better)')
    parser.add_argument('--lkh_time_limit', type=int, default=1,
                        help='LKH time limit per segment')
    parser.add_argument('--lkh_max_trials', type=int, default=100,
                        help='LKH max trials per segment')

    # Checkpoint
    parser.add_argument('--checkpoint', type=str, default='',
                        help='Path to checkpoint to resume from')

    opts = parser.parse_args()

    print("=" * 60)
    print("E(2)-Equivariant CVRP Partition Training")
    print("=" * 60)

    train(opts)


if __name__ == '__main__':
    main()
