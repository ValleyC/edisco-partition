"""
GLOP-Style Training for E(2)-Equivariant CVRP Partition.

This script trains the HeatmapEGNN model using:
1. Sequential sampling (like GLOP) - low variance gradients
2. REINFORCE with baseline - policy gradient optimization
3. E(2)-equivariant network - rotation/translation invariance

Usage:
    python scripts/train_heatmap.py --n_customers 200 --n_epochs 50
    python scripts/train_heatmap.py --n_customers 500 --n_epochs 100 --k_sparse 100
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
import argparse
import time
from tqdm import tqdm
from typing import Tuple, Dict
from dataclasses import dataclass

from edisco_partition.model import HeatmapEGNN, HeatmapConfig, create_heatmap_model
from edisco_partition.sampler import SequentialSampler, compute_route_distance


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class TrainConfig:
    """Training configuration."""
    # Problem
    n_customers: int = 200
    capacity: float = 50.0

    # Model
    n_layers: int = 8
    node_dim: int = 64
    edge_dim: int = 64
    hidden_dim: int = 128
    k_sparse: int = 50

    # Training
    n_epochs: int = 50
    steps_per_epoch: int = 256
    batch_size: int = 10  # Instances per gradient step
    n_samples: int = 10   # Samples per instance for REINFORCE
    lr: float = 3e-4
    weight_decay: float = 1e-5
    grad_clip: float = 1.0

    # Validation
    val_size: int = 100

    # Other
    seed: int = 42
    device: str = 'cuda'
    checkpoint_dir: str = './checkpoints'


# ============================================================================
# DATA GENERATION
# ============================================================================

def generate_instance(n_customers: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Generate random CVRP instance.

    Returns:
        coords: (n_nodes, 2) with depot at index 0
        demands: (n_nodes,) with depot demand = 0
        capacity: Vehicle capacity
    """
    # Capacity scales with problem size
    capacities = {
        100: 50.,
        200: 80.,
        500: 100.,
        1000: 200.,
        2000: 300.,
    }
    capacity = capacities.get(n_customers, 100.)

    n_nodes = n_customers + 1

    # Random coordinates in [0, 1]
    coords = torch.rand(n_nodes, 2, device=device)

    # Random demands in [1, 10]
    demands = torch.randint(1, 10, (n_nodes,), device=device).float()
    demands[0] = 0  # Depot has no demand

    return coords, demands, capacity


# ============================================================================
# ROUTE EVALUATION
# ============================================================================

def evaluate_routes_nn(coords: torch.Tensor, routes: torch.Tensor) -> torch.Tensor:
    """
    Evaluate routes using nearest neighbor within each segment.

    Args:
        coords: (n_nodes, 2)
        routes: (n_samples, seq_len) route sequences

    Returns:
        costs: (n_samples,) total cost per route
    """
    n_samples = routes.size(0)
    device = coords.device
    coords_np = coords.cpu().numpy()
    depot = coords_np[0]

    costs = []

    for i in range(n_samples):
        route = routes[i].cpu().numpy()
        cost = 0.0

        # Split into segments at depot visits
        segments = []
        current_segment = []

        for node in route:
            if node == 0:
                if current_segment:
                    segments.append(current_segment)
                    current_segment = []
            else:
                current_segment.append(node)

        if current_segment:
            segments.append(current_segment)

        # Compute cost for each segment
        for segment in segments:
            if len(segment) == 0:
                continue

            segment_coords = coords_np[segment]
            n = len(segment)

            if n == 1:
                cost += 2 * np.linalg.norm(depot - segment_coords[0])
                continue

            # Nearest neighbor tour
            visited = [False] * n
            current = depot

            for _ in range(n):
                best_dist = float('inf')
                best_idx = -1
                for j in range(n):
                    if not visited[j]:
                        dist = np.linalg.norm(current - segment_coords[j])
                        if dist < best_dist:
                            best_dist = dist
                            best_idx = j
                if best_idx >= 0:
                    visited[best_idx] = True
                    cost += best_dist
                    current = segment_coords[best_idx]

            # Return to depot
            cost += np.linalg.norm(current - depot)

        costs.append(cost)

    return torch.tensor(costs, device=device, dtype=torch.float32)


# ============================================================================
# TRAINING
# ============================================================================

class Trainer:
    """GLOP-style trainer for HeatmapEGNN."""

    def __init__(self, config: TrainConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')

        # Create model
        model_config = HeatmapConfig(
            n_layers=config.n_layers,
            node_dim=config.node_dim,
            edge_dim=config.edge_dim,
            hidden_dim=config.hidden_dim,
            k_sparse=config.k_sparse,
        )
        self.model = HeatmapEGNN(model_config).to(self.device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.n_epochs
        )

        # Tracking
        self.best_val_obj = float('inf')
        self.epoch = 0

    def train_batch(self) -> Tuple[float, float]:
        """Train on a single batch of instances."""
        self.model.train()

        loss_list = []
        obj_list = []

        for _ in range(self.config.batch_size):
            # Generate instance
            coords, demands, capacity = generate_instance(
                self.config.n_customers, self.device
            )

            # Build k-NN graph
            edge_index = HeatmapEGNN.build_knn_graph(coords, self.config.k_sparse)

            # Forward pass - get heatmap
            heatmap = self.model(coords, demands, capacity, edge_index)

            # Normalize heatmap
            heatmap = heatmap / (heatmap.min() + 1e-5)
            heatmap = heatmap + 1e-5

            # Sequential sampling
            sampler = SequentialSampler(
                demands, heatmap, capacity,
                self.config.n_samples, self.device
            )
            routes, log_probs = sampler.sample(require_log_prob=True)

            # Evaluate routes
            costs = evaluate_routes_nn(coords, routes)
            obj_list.append(costs.mean().item())

            # REINFORCE loss with baseline
            baseline = costs.mean()
            total_log_prob = log_probs.sum(dim=1)  # Sum over sequence
            reinforce_loss = ((costs - baseline) * total_log_prob).sum() / self.config.n_samples

            loss_list.append(reinforce_loss)

        # Aggregate loss
        loss = sum(loss_list) / self.config.batch_size

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.grad_clip
        )

        self.optimizer.step()

        return loss.item(), np.mean(obj_list)

    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch."""
        losses = []
        objs = []

        pbar = tqdm(range(self.config.steps_per_epoch), desc=f"Epoch {self.epoch}")

        for step in pbar:
            loss, obj = self.train_batch()
            losses.append(loss)
            objs.append(obj)

            pbar.set_postfix({
                'loss': f'{loss:.4f}',
                'obj': f'{obj:.2f}',
                'avg': f'{np.mean(objs):.2f}'
            })

        self.scheduler.step()

        return np.mean(losses), np.mean(objs)

    @torch.no_grad()
    def validate(self) -> float:
        """Validate on random instances."""
        self.model.eval()
        objs = []

        for _ in tqdm(range(self.config.val_size), desc="Validation", leave=False):
            # Generate instance
            coords, demands, capacity = generate_instance(
                self.config.n_customers, self.device
            )

            # Build graph
            edge_index = HeatmapEGNN.build_knn_graph(coords, self.config.k_sparse)

            # Get heatmap
            heatmap = self.model(coords, demands, capacity, edge_index)
            heatmap = heatmap / (heatmap.min() + 1e-5)
            heatmap = heatmap + 1e-5

            # Greedy sampling
            sampler = SequentialSampler(demands, heatmap, capacity, 1, self.device)
            routes, _ = sampler.sample(require_log_prob=False, greedy=True)

            # Evaluate
            cost = evaluate_routes_nn(coords, routes)
            objs.append(cost.item())

        return np.mean(objs)

    @torch.no_grad()
    def check_equivariance(self) -> float:
        """Verify rotation equivariance."""
        self.model.eval()

        # Generate instance
        coords, demands, capacity = generate_instance(
            min(self.config.n_customers, 100), self.device
        )
        edge_index = HeatmapEGNN.build_knn_graph(coords, self.config.k_sparse)

        # Original heatmap
        heatmap1 = self.model(coords, demands, capacity, edge_index)

        # Rotate 90 degrees
        angle = torch.tensor(np.pi / 2, device=self.device)
        cos_t, sin_t = torch.cos(angle), torch.sin(angle)
        rot = torch.tensor([[cos_t, -sin_t], [sin_t, cos_t]], device=self.device, dtype=torch.float32)
        coords_rot = coords @ rot.T

        edge_index_rot = HeatmapEGNN.build_knn_graph(coords_rot, self.config.k_sparse)
        heatmap2 = self.model(coords_rot, demands, capacity, edge_index_rot)

        # Compare
        diff = (heatmap1 - heatmap2).abs().max().item()
        return diff

    def save_checkpoint(self, path: str, is_best: bool = False):
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True)

        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_obj': self.best_val_obj,
            'config': self.config,
        }
        torch.save(checkpoint, path)

        if is_best:
            best_path = path.replace('.pt', '_best.pt')
            torch.save(checkpoint, best_path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_val_obj = checkpoint.get('best_val_obj', float('inf'))

    def train(self):
        """Main training loop."""
        print("=" * 60)
        print("EDISCO-Partition: GLOP-Style Training")
        print("=" * 60)
        print(f"Customers: {self.config.n_customers}")
        print(f"k_sparse: {self.config.k_sparse}")
        print(f"Network: {self.config.n_layers} layers, {self.config.node_dim} dim")
        print(f"Training: {self.config.n_epochs} epochs, {self.config.steps_per_epoch} steps/epoch")
        print(f"Samples per instance: {self.config.n_samples}")
        print(f"Device: {self.device}")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print("=" * 60)

        # Initial equivariance check
        print("\nVerifying equivariance...")
        diff = self.check_equivariance()
        status = "PASS" if diff < 0.01 else "FAIL"
        print(f"  90° rotation max diff: {diff:.6f} [{status}]")

        # Initial validation
        print("\nInitial validation...")
        val_obj = self.validate()
        print(f"Epoch 0: val_obj={val_obj:.4f}")
        self.best_val_obj = val_obj

        # Training loop
        print("\nStarting training...")
        total_time = 0

        for epoch in range(1, self.config.n_epochs + 1):
            self.epoch = epoch
            start_time = time.time()

            # Train
            train_loss, train_obj = self.train_epoch()
            epoch_time = time.time() - start_time
            total_time += epoch_time

            # Validate
            val_obj = self.validate()

            # Learning rate
            lr = self.optimizer.param_groups[0]['lr']

            print(f"Epoch {epoch}: val={val_obj:.4f}, train={train_obj:.4f}, "
                  f"loss={train_loss:.4f}, lr={lr:.2e}, time={epoch_time:.1f}s")

            # Save checkpoint
            is_best = val_obj < self.best_val_obj
            if is_best:
                self.best_val_obj = val_obj
                print(f"  -> New best!")

            checkpoint_path = os.path.join(
                self.config.checkpoint_dir,
                f"heatmap_cvrp{self.config.n_customers}_epoch{epoch}.pt"
            )
            self.save_checkpoint(checkpoint_path, is_best)

            # Periodic equivariance check
            if epoch % 5 == 0:
                diff = self.check_equivariance()
                status = "PASS" if diff < 0.01 else "FAIL"
                print(f"  Equivariance: max_diff={diff:.6f} [{status}]")

        print("\n" + "=" * 60)
        print("Training complete!")
        print(f"Total time: {total_time:.1f}s ({total_time/60:.1f}min)")
        print(f"Best validation: {self.best_val_obj:.4f}")
        print("=" * 60)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train HeatmapEGNN with GLOP-style REINFORCE')

    # Problem
    parser.add_argument('--n_customers', type=int, default=200)

    # Model
    parser.add_argument('--n_layers', type=int, default=8)
    parser.add_argument('--node_dim', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--k_sparse', type=int, default=50)

    # Training
    parser.add_argument('--n_epochs', type=int, default=50)
    parser.add_argument('--steps_per_epoch', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--n_samples', type=int, default=10)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--grad_clip', type=float, default=1.0)

    # Validation
    parser.add_argument('--val_size', type=int, default=100)

    # Other
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--resume', type=str, default='', help='Path to checkpoint to resume')

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Create config
    config = TrainConfig(
        n_customers=args.n_customers,
        n_layers=args.n_layers,
        node_dim=args.node_dim,
        hidden_dim=args.hidden_dim,
        k_sparse=args.k_sparse,
        n_epochs=args.n_epochs,
        steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size,
        n_samples=args.n_samples,
        lr=args.lr,
        grad_clip=args.grad_clip,
        val_size=args.val_size,
        seed=args.seed,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
    )

    # Create trainer
    trainer = Trainer(config)

    # Resume if specified
    if args.resume:
        print(f"Resuming from {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Train
    trainer.train()


if __name__ == '__main__':
    main()
