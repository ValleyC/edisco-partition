# EDISCO-Partition

**E(2)-Equivariant Partition Network for Large-Scale Capacitated Vehicle Routing Problem**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

EDISCO-Partition is a novel approach to solving large-scale CVRP instances using **E(2)-equivariant graph neural networks** for partitioning. Unlike existing methods (e.g., GLOP) that use polar coordinates which break rotational symmetry, our approach maintains **provable equivariance** under:

- **Rotation**: Rotating the problem produces consistently rotated cluster assignments
- **Translation**: Shifting coordinates doesn't change cluster assignments
- **Reflection**: Mirroring the instance produces consistently mirrored assignments

This leads to more robust and generalizable partitioning that doesn't depend on arbitrary coordinate frame choices.

## Key Features

- **E(2)-Equivariant Architecture**: EGNN-based encoder with strict equivariance guarantees
- **Scalable**: Designed for CVRP-200, CVRP-500, and CVRP-1000
- **Multiple Routing Methods**: Nearest Neighbor, 2-opt improvement, EDISCO integration
- **Comprehensive Baselines**: K-Means, Sweep, Polar (GLOP-style), Random
- **Production Ready**: Mixed precision training, gradient accumulation, checkpointing

## Installation

### From Source

```bash
git clone https://github.com/yourusername/edisco-partition.git
cd edisco-partition
pip install -e .
```

### With Optional Dependencies

```bash
# With visualization support
pip install -e ".[visualization]"

# With logging (TensorBoard, WandB)
pip install -e ".[logging]"

# All optional dependencies
pip install -e ".[all]"
```

## Quick Start

### Solving a Random Instance

```python
from edisco_partition import EDISCOInferencePipeline, RandomCVRPGenerator

# Generate random CVRP instance
instance = RandomCVRPGenerator.generate(
    n_customers=200,
    distribution='uniform',
    seed=42
)

# Create inference pipeline
pipeline = EDISCOInferencePipeline(
    checkpoint_path='checkpoints/best_model.pt',  # Optional
    n_clusters=10,
    routing_method='two_opt',
    device='cuda'
)

# Solve
solution = pipeline.solve(instance)
print(f"Total distance: {solution['distance']:.4f}")
print(f"Number of routes: {solution['n_routes']}")
```

### Command Line Interface

```bash
# Solve from file
python -m scripts.train --input instance.pkl --checkpoint best_model.pt

# Generate and solve random instance
python -m edisco_partition.inference --random --n_customers 500 --visualize

# Train a new model
python scripts/train.py --n_customers 200 --epochs 100 --batch_size 32

# Evaluate on benchmarks
python scripts/evaluate.py --checkpoint best_model.pt --all_sizes
```

## Architecture

```
Input: coords (B, N, 2), demands (B, N), capacity (B,)
  │
  ├─► NodeEmbedding: demands → invariant features
  │
  ├─► MultiScaleEGNN:
  │     └─► L EGNN layers with attention
  │         - Equivariant coordinate updates
  │         - Invariant feature updates
  │         - Skip connections
  │
  ├─► CapacityAwareClusterHead:
  │     └─► Demand-aware cluster logits
  │
  └─► Output: cluster_logits (B, N, K)
```

### Why E(2)-Equivariance Matters

Traditional partition methods like GLOP use polar coordinates (r, θ) which are **not equivariant**:
- Rotating the input changes the θ values
- This can lead to completely different partitions for rotated versions of the same problem

Our EGNN-based approach:
- Uses only **relative positions** (distances) which are rotation-invariant
- Produces **consistent** partitions regardless of coordinate frame orientation
- Achieves better generalization across different problem orientations

## Training

### Configuration

```python
from edisco_partition import PartitionConfig

config = PartitionConfig(
    n_clusters=10,           # Number of clusters
    n_layers=8,              # EGNN depth
    hidden_dim=256,          # Feature dimension
    n_heads=8,               # Attention heads
    dropout=0.1,

    # Loss weights
    balance_weight=1.0,      # Cluster size balance
    compactness_weight=0.5,  # Spatial compactness
    entropy_weight=0.1,      # Assignment confidence
    supervised_weight=1.0    # Match ground truth
)
```

### Training a Model

```bash
python scripts/train.py \
    --n_customers 200 \
    --n_clusters 10 \
    --epochs 100 \
    --batch_size 32 \
    --lr 1e-4 \
    --save_dir checkpoints/cvrp200
```

## Evaluation

### Running Benchmarks

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/best_model.pt \
    --problem_sizes 200 500 1000 \
    --distributions uniform clustered mixed \
    --n_instances 100 \
    --output_dir results/
```

### Baseline Comparison

The evaluation framework includes:
- **K-Means**: Standard clustering baseline
- **Sweep**: Classic VRP heuristic (angle-based)
- **Polar Sector**: GLOP-style polar coordinate partitioning
- **Polar K-Means**: K-means in polar coordinate space
- **Random**: Lower bound reference

## Project Structure

```
edisco-partition/
├── edisco_partition/          # Main package
│   ├── __init__.py           # Package exports
│   ├── model.py              # E(2)-Equivariant Partition Network
│   ├── solver.py             # Routing integration & baselines
│   ├── data.py               # Data generation utilities
│   └── inference.py          # End-to-end inference pipeline
├── scripts/
│   ├── train.py              # Training script
│   └── evaluate.py           # Evaluation script
├── configs/                   # Configuration files
├── examples/                  # Example notebooks
├── setup.py
├── pyproject.toml
├── requirements.txt
└── README.md
```

## API Reference

### Core Classes

| Class | Description |
|-------|-------------|
| `EquivariantPartitionNet` | E(2)-equivariant partition network |
| `EDISCOPartitionSolver` | Complete partition + routing solver |
| `EDISCOInferencePipeline` | End-to-end inference pipeline |
| `LargeCVRPDataset` | PyTorch dataset for CVRP instances |

### Key Functions

| Function | Description |
|----------|-------------|
| `verify_partition_equivariance()` | Verify E(2)-equivariance of model |
| `generate_cvrp_dataset()` | Generate CVRP benchmark data |
| `compare_solvers()` | Compare multiple partition methods |

## Citation

If you use this code in your research, please cite:

```bibtex
@article{edisco2025,
  title={EDISCO: Equivariant Diffusion for Combinatorial Optimization},
  author={...},
  journal={...},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This work builds on the EDISCO framework for combinatorial optimization
- EGNN architecture inspired by [Satorras et al., 2021](https://arxiv.org/abs/2102.09844)
