"""
Training Pipeline for EDISCO-Partition
======================================

Complete training pipeline for the E(2)-Equivariant Partition Network.

Features:
- Multi-GPU training support
- Gradient accumulation
- Learning rate scheduling
- Comprehensive logging (TensorBoard, WandB)
- Checkpointing with best model tracking
- Evaluation during training
- Data augmentation

Usage:
    python train_partition.py --config configs/cvrp200.yaml
    python train_partition.py --n_customers 500 --epochs 100

Author: EDISCO Team
"""

import os
import sys
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

# Package imports
from edisco_partition import (
    EquivariantPartitionNet,
    PartitionConfig,
    PartitionLoss,
    verify_partition_equivariance,
    compute_route_distance,
    EDISCOPartitionSolver,
    SolverConfig,
    RoutingMethod,
    KMeansPartitionSolver,
    SweepPartitionSolver,
    compare_solvers,
    LargeCVRPDataset,
    cvrp_collate_fn,
    generate_cvrp_dataset,
    DistributionType
)
from edisco_partition.model import ReinforceLoss


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class TrainingConfig:
    """Complete training configuration."""
    # Data
    n_customers: int = 200
    train_size: int = 10000
    val_size: int = 1000
    test_size: int = 1000
    distribution: str = "uniform"
    data_dir: str = "./data/cvrp_large"
    solver: str = "heuristic"  # 'heuristic' (fast), 'lkh' (slow but optimal), 'auto'

    # Model
    n_clusters: int = 10
    n_layers: int = 8
    hidden_dim: int = 256
    node_dim: int = 128
    n_heads: int = 8
    dropout: float = 0.1

    # Loss configuration
    loss_type: str = "reinforce"  # 'reinforce' (direct) or 'multi' (multiple components)
    n_samples: int = 8  # Number of samples for REINFORCE (GLOP uses 10, need >1 for self-critical baseline)

    # Loss weights (only used for 'multi' loss type)
    balance_weight: float = 1.0
    compactness_weight: float = 0.5
    entropy_weight: float = 0.1
    supervised_weight: float = 1.0
    coverage_weight: float = 0.5

    # Training
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_epochs: int = 5
    grad_clip: float = 1.0
    grad_accumulation: int = 1

    # Mixed precision
    use_amp: bool = True

    # Data augmentation
    augment: bool = True

    # Evaluation
    eval_every: int = 5
    eval_batch_size: int = 16

    # Checkpointing
    save_dir: str = "./checkpoints/partition"
    save_every: int = 10

    # Logging
    log_dir: str = "./logs/partition"
    use_wandb: bool = False
    wandb_project: str = "edisco-partition"

    # Hardware
    device: str = "cuda"
    num_workers: int = 4
    seed: int = 42

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> 'TrainingConfig':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_yaml(cls, path: str) -> 'TrainingConfig':
        import yaml
        with open(path, 'r') as f:
            d = yaml.safe_load(f)
        return cls.from_dict(d)


# ============================================================================
# LOGGER
# ============================================================================

class TrainingLogger:
    """Comprehensive logging for training."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.log_dir = Path(config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.tb_writer = SummaryWriter(str(self.log_dir))
        except ImportError:
            self.tb_writer = None
            warnings.warn("TensorBoard not available")

        # Weights & Biases
        self.wandb_run = None
        if config.use_wandb:
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=config.wandb_project,
                    config=config.to_dict(),
                    dir=str(self.log_dir)
                )
            except Exception as e:
                warnings.warn(f"WandB initialization failed: {e}")

        # Metrics storage
        self.train_losses = []
        self.val_losses = []
        self.val_gaps = []

    def log_scalar(self, tag: str, value: float, step: int):
        """Log a scalar value."""
        if self.tb_writer:
            self.tb_writer.add_scalar(tag, value, step)
        if self.wandb_run:
            import wandb
            wandb.log({tag: value}, step=step)

    def log_scalars(self, main_tag: str, scalars: Dict[str, float], step: int):
        """Log multiple scalars."""
        if self.tb_writer:
            self.tb_writer.add_scalars(main_tag, scalars, step)
        if self.wandb_run:
            import wandb
            wandb.log({f"{main_tag}/{k}": v for k, v in scalars.items()}, step=step)

    def log_histogram(self, tag: str, values: torch.Tensor, step: int):
        """Log histogram."""
        if self.tb_writer:
            self.tb_writer.add_histogram(tag, values, step)

    def log_epoch(self, epoch: int, train_loss: float, val_loss: Optional[float] = None,
                  val_gap: Optional[float] = None, lr: Optional[float] = None):
        """Log epoch summary."""
        self.train_losses.append(train_loss)

        log_str = f"Epoch {epoch}: train_loss={train_loss:.4f}"

        if val_loss is not None:
            self.val_losses.append(val_loss)
            log_str += f", val_loss={val_loss:.4f}"

        if val_gap is not None:
            self.val_gaps.append(val_gap)
            log_str += f", val_gap={val_gap:.2f}%"

        if lr is not None:
            log_str += f", lr={lr:.2e}"

        print(log_str)

        # Log to backends
        self.log_scalar("train/loss", train_loss, epoch)
        if val_loss is not None:
            self.log_scalar("val/loss", val_loss, epoch)
        if val_gap is not None:
            self.log_scalar("val/gap", val_gap, epoch)
        if lr is not None:
            self.log_scalar("train/lr", lr, epoch)

    def close(self):
        """Close logging backends."""
        if self.tb_writer:
            self.tb_writer.close()
        if self.wandb_run:
            import wandb
            wandb.finish()


# ============================================================================
# LEARNING RATE SCHEDULER
# ============================================================================

class WarmupCosineScheduler:
    """Learning rate scheduler with warmup and cosine decay."""

    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float = 1e-6
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

    def step(self, epoch: int):
        """Update learning rate."""
        if epoch < self.warmup_epochs:
            # Linear warmup
            scale = (epoch + 1) / self.warmup_epochs
        else:
            # Cosine decay
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            scale = 0.5 * (1 + np.cos(np.pi * progress))

        for i, group in enumerate(self.optimizer.param_groups):
            group['lr'] = max(self.min_lr, self.base_lrs[i] * scale)

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']


# ============================================================================
# TRAINER
# ============================================================================

class PartitionTrainer:
    """
    Complete trainer for the partition network.
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')

        # Set seed
        self._set_seed(config.seed)

        # Create model
        self.model = self._create_model()
        self.model = self.model.to(self.device)

        # Create optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # Create scheduler
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            config.warmup_epochs,
            config.epochs
        )

        # Mixed precision
        self.scaler = GradScaler() if config.use_amp else None

        # Loss function
        partition_config = PartitionConfig(
            n_clusters=config.n_clusters,
            balance_weight=config.balance_weight,
            compactness_weight=config.compactness_weight,
            entropy_weight=config.entropy_weight,
            supervised_weight=config.supervised_weight,
            coverage_weight=config.coverage_weight
        )

        if config.loss_type == 'reinforce':
            self.loss_fn = ReinforceLoss(partition_config)
            print("Using REINFORCE loss (direct distance optimization)")
        else:
            self.loss_fn = PartitionLoss(partition_config)
            print("Using multi-component loss")

        # Logger
        self.logger = TrainingLogger(config)

        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_val_gap = float('inf')
        self.best_epoch = 0

        # Create directories
        Path(config.save_dir).mkdir(parents=True, exist_ok=True)

    def _set_seed(self, seed: int):
        """Set random seeds for reproducibility."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _create_model(self) -> EquivariantPartitionNet:
        """Create the partition model."""
        config = PartitionConfig(
            n_clusters=self.config.n_clusters,
            n_layers=self.config.n_layers,
            hidden_dim=self.config.hidden_dim,
            node_dim=self.config.node_dim,
            n_heads=self.config.n_heads,
            dropout=self.config.dropout
        )
        return EquivariantPartitionNet(config)

    def _prepare_data(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Prepare data loaders."""
        data_dir = Path(self.config.data_dir)
        distribution = DistributionType(self.config.distribution)

        # Check for existing data
        train_path = data_dir / f"cvrp_{self.config.n_customers}_{self.config.distribution}_train.pkl.gz"
        val_path = data_dir / f"cvrp_{self.config.n_customers}_{self.config.distribution}_val.pkl.gz"
        test_path = data_dir / f"cvrp_{self.config.n_customers}_{self.config.distribution}_test.pkl.gz"

        # Generate data if needed (use heuristics for speed, LKH for quality)
        solver = getattr(self.config, 'solver', 'heuristic')  # Default to fast heuristics

        if not train_path.exists():
            print(f"Generating training data (solver={solver})...")
            generate_cvrp_dataset(
                n_instances=self.config.train_size,
                n_customers=self.config.n_customers,
                output_path=str(train_path).replace('.gz', ''),
                distribution=distribution,
                solver=solver,
                time_limit=30,
                base_seed=self.config.seed
            )

        if not val_path.exists():
            print(f"Generating validation data (solver={solver})...")
            generate_cvrp_dataset(
                n_instances=self.config.val_size,
                n_customers=self.config.n_customers,
                output_path=str(val_path).replace('.gz', ''),
                distribution=distribution,
                solver=solver,
                time_limit=60,
                base_seed=self.config.seed + 100000
            )

        if not test_path.exists():
            print(f"Generating test data (solver={solver})...")
            generate_cvrp_dataset(
                n_instances=self.config.test_size,
                n_customers=self.config.n_customers,
                output_path=str(test_path).replace('.gz', ''),
                distribution=distribution,
                solver=solver,
                time_limit=120,
                base_seed=self.config.seed + 200000
            )

        # Load datasets
        train_dataset = LargeCVRPDataset(
            str(train_path) if train_path.exists() else str(train_path).replace('.gz', '') + '.gz',
            augment=self.config.augment,
            return_gt_clusters=True
        )

        val_dataset = LargeCVRPDataset(
            str(val_path) if val_path.exists() else str(val_path).replace('.gz', '') + '.gz',
            augment=False,
            return_gt_clusters=True
        )

        test_dataset = LargeCVRPDataset(
            str(test_path) if test_path.exists() else str(test_path).replace('.gz', '') + '.gz',
            augment=False,
            return_gt_clusters=True
        )

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            collate_fn=cvrp_collate_fn,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.eval_batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=cvrp_collate_fn,
            pin_memory=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.eval_batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=cvrp_collate_fn,
            pin_memory=True
        )

        return train_loader, val_loader, test_loader

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        loss_components = {}
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for batch_idx, batch in enumerate(pbar):
            # Move to device
            coords = batch['coords'].to(self.device)
            demands = batch['demands'].to(self.device)
            capacity = batch['capacity'].to(self.device)

            gt_clusters = batch.get('gt_clusters')
            if gt_clusters is not None:
                gt_clusters = gt_clusters.to(self.device)

            # Forward pass with mixed precision
            with autocast(enabled=self.config.use_amp):
                outputs = self.model(coords, demands, capacity)

                # Different loss interfaces
                if self.config.loss_type == 'reinforce':
                    loss, loss_dict = self.loss_fn(
                        outputs, coords, demands, capacity,
                        n_samples=self.config.n_samples
                    )
                else:
                    loss, loss_dict = self.loss_fn(
                        outputs, coords, demands, capacity, gt_clusters
                    )

                # Scale for gradient accumulation
                loss = loss / self.config.grad_accumulation

            # Backward pass
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Update weights
            if (batch_idx + 1) % self.config.grad_accumulation == 0:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.grad_clip
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.grad_clip
                    )
                    self.optimizer.step()

                self.optimizer.zero_grad()

            # Accumulate metrics
            total_loss += loss.item() * self.config.grad_accumulation
            for k, v in loss_dict.items():
                loss_components[k] = loss_components.get(k, 0) + v
            n_batches += 1

            # Update progress bar
            if self.config.loss_type == 'reinforce' and 'distance' in loss_dict:
                pbar.set_postfix({
                    'dist': f"{loss_dict.get('distance', 0):.2f}",
                    'loss': f"{loss_dict.get('total', 0):.4f}"
                })
            else:
                pbar.set_postfix({'loss': f"{total_loss / n_batches:.4f}"})

        # Average metrics
        avg_loss = total_loss / n_batches
        for k in loss_components:
            loss_components[k] /= n_batches

        return {'loss': avg_loss, **loss_components}

    @torch.no_grad()
    def evaluate(
        self,
        val_loader: DataLoader,
        compute_gap: bool = True
    ) -> Dict[str, float]:
        """Evaluate on validation set."""
        self.model.eval()

        total_loss = 0.0
        loss_components = {}
        n_batches = 0

        all_distances = []
        gt_distances = []

        for batch in tqdm(val_loader, desc="Evaluating"):
            coords = batch['coords'].to(self.device)
            demands = batch['demands'].to(self.device)
            capacity = batch['capacity'].to(self.device)

            gt_clusters = batch.get('gt_clusters')
            if gt_clusters is not None:
                gt_clusters = gt_clusters.to(self.device)

            # Forward pass
            outputs = self.model(coords, demands, capacity)

            # Different loss interfaces
            if self.config.loss_type == 'reinforce':
                loss, loss_dict = self.loss_fn(
                    outputs, coords, demands, capacity, greedy=True
                )
            else:
                loss, loss_dict = self.loss_fn(
                    outputs, coords, demands, capacity, gt_clusters
                )

            total_loss += loss.item()
            for k, v in loss_dict.items():
                loss_components[k] = loss_components.get(k, 0) + v
            n_batches += 1

            # Compute actual routing distances if needed
            if compute_gap and 'gt_distance' in batch:
                gt_distances.extend(batch['gt_distance'].tolist())

                # Get clusters and route
                clusters = self.model.get_clusters(
                    outputs['cluster_logits'],
                    demands,
                    capacity[0].item()
                )

                for b in range(coords.shape[0]):
                    coords_np = coords[b].cpu().numpy()
                    demands_np = demands[b].cpu().numpy()
                    cap = capacity[b].item()

                    # Route clusters
                    routes = []
                    for cluster in clusters[b]:
                        if not cluster:
                            continue

                        cluster_demand = sum(demands_np[c] for c in cluster)
                        if cluster_demand <= cap:
                            # Simple nearest neighbor routing
                            route = self._nn_route(coords_np, cluster)
                            routes.append(route)
                        else:
                            # Split and route
                            sub_routes = self._split_and_route(
                                coords_np, cluster, demands_np, cap
                            )
                            routes.extend(sub_routes)

                    dist = compute_route_distance(coords_np, routes)
                    all_distances.append(dist)

        # Average metrics
        avg_loss = total_loss / n_batches
        for k in loss_components:
            loss_components[k] /= n_batches

        result = {'loss': avg_loss, **loss_components}

        # Compute gap
        if compute_gap and all_distances and gt_distances:
            gaps = [(d - g) / g * 100 for d, g in zip(all_distances, gt_distances)]
            result['gap_mean'] = np.mean(gaps)
            result['gap_std'] = np.std(gaps)
            result['distance_mean'] = np.mean(all_distances)

        return result

    def _nn_route(self, coords: np.ndarray, customers: List[int]) -> List[int]:
        """Nearest neighbor routing."""
        if len(customers) <= 1:
            return customers

        route = []
        remaining = set(customers)
        current = coords[0]

        while remaining:
            nearest = min(remaining, key=lambda c: np.linalg.norm(current - coords[c]))
            route.append(nearest)
            current = coords[nearest]
            remaining.remove(nearest)

        return route

    def _split_and_route(
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
                    routes.append(self._nn_route(coords, current))
                current = [c]
                current_load = demands[c]

        if current:
            routes.append(self._nn_route(coords, current))

        return routes

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.to_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_gap': self.best_val_gap
        }

        # Save regular checkpoint
        path = Path(self.config.save_dir) / f"checkpoint_epoch{epoch}.pt"
        torch.save(checkpoint, path)

        # Save best model
        if is_best:
            best_path = Path(self.config.save_dir) / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"Saved best model (epoch {epoch})")

        # Save latest
        latest_path = Path(self.config.save_dir) / "latest.pt"
        torch.save(checkpoint, latest_path)

    def load_checkpoint(self, path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_val_gap = checkpoint.get('best_val_gap', float('inf'))

        return checkpoint.get('epoch', 0)

    def train(self, resume_from: Optional[str] = None):
        """Main training loop."""
        print("=" * 60)
        print("EDISCO-Partition Training")
        print("=" * 60)
        print(f"Config: {self.config.n_customers} customers, "
              f"{self.config.n_clusters} clusters")
        print(f"Model: {self.config.n_layers} layers, "
              f"{self.config.hidden_dim} hidden dim")
        print(f"Device: {self.device}")
        print("=" * 60)

        # Verify equivariance
        print("\nVerifying model equivariance...")
        eq_results = verify_partition_equivariance(
            self.model,
            batch_size=2,
            n_nodes=self.config.n_customers + 1,
            device=str(self.device)
        )
        for test, passed in eq_results.items():
            status = "PASSED" if passed else "FAILED"
            print(f"  {test}: {status}")

        # Prepare data
        print("\nPreparing data...")
        train_loader, val_loader, test_loader = self._prepare_data()
        print(f"Train: {len(train_loader.dataset)} instances")
        print(f"Val: {len(val_loader.dataset)} instances")
        print(f"Test: {len(test_loader.dataset)} instances")

        # Resume from checkpoint
        start_epoch = 0
        if resume_from:
            print(f"\nResuming from {resume_from}")
            start_epoch = self.load_checkpoint(resume_from) + 1

        # Training loop
        print("\nStarting training...")
        for epoch in range(start_epoch, self.config.epochs):
            # Update learning rate
            self.scheduler.step(epoch)
            current_lr = self.scheduler.get_lr()

            # Train
            train_metrics = self.train_epoch(train_loader, epoch)

            # Evaluate
            val_metrics = None
            if (epoch + 1) % self.config.eval_every == 0:
                val_metrics = self.evaluate(val_loader, compute_gap=True)

                # Check for best model
                is_best = False
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    is_best = True

                if 'gap_mean' in val_metrics and val_metrics['gap_mean'] < self.best_val_gap:
                    self.best_val_gap = val_metrics['gap_mean']
                    self.best_epoch = epoch
                    is_best = True

                # Save checkpoint
                if is_best or (epoch + 1) % self.config.save_every == 0:
                    self.save_checkpoint(epoch, is_best)

            # Log
            self.logger.log_epoch(
                epoch,
                train_metrics['loss'],
                val_metrics['loss'] if val_metrics else None,
                val_metrics.get('gap_mean') if val_metrics else None,
                current_lr
            )

            # Log detailed metrics
            self.logger.log_scalars("train/loss_components", train_metrics, epoch)
            if val_metrics:
                self.logger.log_scalars("val/metrics", val_metrics, epoch)

        # Final evaluation on test set
        print("\n" + "=" * 60)
        print("Final Evaluation on Test Set")
        print("=" * 60)

        # Load best model
        best_path = Path(self.config.save_dir) / "best_model.pt"
        if best_path.exists():
            self.load_checkpoint(str(best_path))
            print(f"Loaded best model from epoch {self.best_epoch}")

        test_metrics = self.evaluate(test_loader, compute_gap=True)
        print(f"Test Loss: {test_metrics['loss']:.4f}")
        if 'gap_mean' in test_metrics:
            print(f"Test Gap: {test_metrics['gap_mean']:.2f}% +/- {test_metrics['gap_std']:.2f}%")

        # Compare with baselines
        print("\nComparing with baselines...")
        self._compare_with_baselines(test_loader)

        # Close logger
        self.logger.close()

        print("\nTraining complete!")
        return test_metrics

    def _compare_with_baselines(self, test_loader: DataLoader):
        """Compare with baseline methods."""
        # Get a batch for comparison
        batch = next(iter(test_loader))
        coords = batch['coords'].to(self.device)
        demands = batch['demands'].to(self.device)
        capacity = batch['capacity'].to(self.device)
        gt_distances = batch.get('gt_distance')

        # Create solver wrapper
        solver_config = SolverConfig(
            n_clusters=self.config.n_clusters,
            partition_n_layers=self.config.n_layers,
            partition_hidden_dim=self.config.hidden_dim,
            routing_method=RoutingMethod.TWO_OPT,
            device=str(self.device)
        )

        # Create solver with trained partition network
        solver = EDISCOPartitionSolver(solver_config)
        solver.partition_net = self.model
        solver = solver.to(self.device)

        # Compare
        gt_dist = gt_distances.mean().item() if gt_distances is not None else None
        results = compare_solvers(coords, demands, capacity, solver, gt_dist)

        print("\nBaseline Comparison:")
        for name, metrics in results.items():
            print(f"\n{name}:")
            print(f"  Distance: {metrics['distance']:.3f}")
            print(f"  Time: {metrics['time']:.3f}s")
            if 'gap' in metrics:
                print(f"  Gap: {metrics['gap']:.2f}%")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train EDISCO-Partition")

    # Data
    parser.add_argument('--n_customers', type=int, default=200)
    parser.add_argument('--train_size', type=int, default=10000)
    parser.add_argument('--val_size', type=int, default=1000)
    parser.add_argument('--distribution', type=str, default='uniform')
    parser.add_argument('--data_dir', type=str, default='./data/cvrp_large')
    parser.add_argument('--solver', type=str, default='heuristic',
                       choices=['heuristic', 'lkh', 'auto'],
                       help='Solver for ground truth: heuristic (fast), lkh (slow), auto')

    # Model
    parser.add_argument('--n_clusters', type=int, default=10)
    parser.add_argument('--n_layers', type=int, default=8)
    parser.add_argument('--hidden_dim', type=int, default=256)

    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--grad_clip', type=float, default=1.0)

    # Loss
    parser.add_argument('--loss_type', type=str, default='reinforce',
                       choices=['reinforce', 'multi'],
                       help='Loss type: reinforce (direct distance) or multi (weighted components)')
    parser.add_argument('--n_samples', type=int, default=8,
                       help='Number of samples for REINFORCE (need >1 for self-critical baseline)')

    # Other
    parser.add_argument('--save_dir', type=str, default='./checkpoints/partition')
    parser.add_argument('--log_dir', type=str, default='./logs/partition')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--config', type=str, default=None)

    args = parser.parse_args()

    # Load config from file or create from args
    if args.config:
        config = TrainingConfig.from_yaml(args.config)
    else:
        config = TrainingConfig(
            n_customers=args.n_customers,
            train_size=args.train_size,
            val_size=args.val_size,
            distribution=args.distribution,
            data_dir=args.data_dir,
            solver=args.solver,
            n_clusters=args.n_clusters,
            n_layers=args.n_layers,
            hidden_dim=args.hidden_dim,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            grad_clip=args.grad_clip,
            loss_type=args.loss_type,
            n_samples=args.n_samples,
            save_dir=args.save_dir,
            log_dir=args.log_dir,
            seed=args.seed,
            device=args.device
        )

    # Create trainer and train
    trainer = PartitionTrainer(config)
    trainer.train(resume_from=args.resume)


if __name__ == "__main__":
    main()
