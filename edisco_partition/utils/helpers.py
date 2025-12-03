"""
Utility functions for EDISCO-Partition.
"""

import torch
import numpy as np
import random


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For deterministic behavior (may slow down training)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device_str='auto'):
    """
    Get torch device.

    Args:
        device_str: 'auto', 'cuda', 'cuda:0', 'cpu', etc.

    Returns:
        torch.device
    """
    if device_str == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_str)


def count_parameters(model):
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(model, optimizer, scheduler, epoch, path, **kwargs):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        **kwargs
    }
    torch.save(checkpoint, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    """Load training checkpoint."""
    checkpoint = torch.load(path, map_location='cpu')

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return checkpoint.get('epoch', 0)


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def check_equivariance(model, coords, demand, capacity, k_sparse, device, angle_deg=45):
    """
    Verify E(2)-equivariance of the model.

    Rotates input and checks if output is consistent.

    Args:
        model: PartitionNet
        coords, demand, capacity: CVRP instance
        k_sparse: Graph sparsity
        device: Device
        angle_deg: Rotation angle in degrees

    Returns:
        diff: Mean absolute difference (should be ~0 for perfect equivariance)
    """
    from edisco_partition.data.graph import build_graph

    model.eval()

    # Original
    graph1 = build_graph(coords, demand, capacity, k_sparse)
    with torch.no_grad():
        heatmap1 = model(graph1)

    # Rotate coordinates
    angle = np.radians(angle_deg)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rotation_matrix = torch.tensor([
        [cos_a, -sin_a],
        [sin_a, cos_a]
    ], device=device, dtype=coords.dtype)

    # Rotate around depot
    depot = coords[0:1]
    centered = coords - depot
    rotated = centered @ rotation_matrix.T
    coords_rotated = rotated + depot

    # Build graph with rotated coordinates
    graph2 = build_graph(coords_rotated, demand, capacity, k_sparse)
    with torch.no_grad():
        heatmap2 = model(graph2)

    # Compare heatmaps (should be identical for equivariant model)
    diff = (heatmap1 - heatmap2).abs().mean().item()

    return diff
