# EDISCO-Partition: E(2)-Equivariant Neural Partitioning for Large-Scale CVRP

This repository implements an E(2)-equivariant graph neural network for partitioning large-scale Capacitated Vehicle Routing Problems (CVRP).

## Key Features

- **E(2)-Equivariance**: The model respects rotation and translation symmetries, providing better generalization across problem instances.
- **EGNN Architecture**: Uses E(n)-Equivariant Graph Neural Networks that process coordinates equivariantly.
- **REINFORCE Training**: Policy gradient optimization directly on routing cost.
- **Scalable**: Designed for large-scale CVRP (500-2000+ customers).

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/edisco-partition.git
cd edisco-partition

# Install dependencies
pip install -r requirements.txt

# Optional: Install PyTorch Geometric for GLOP revisers
pip install torch-scatter torch-sparse torch-geometric
```

## Quick Start

### Training

```bash
# Train on CVRP-200
python scripts/train.py --problem_size 200 --n_epochs 50

# Train on CVRP-1000
python scripts/train.py --problem_size 1000 --n_epochs 100 --batch_size 5

# Train with LKH evaluation (slower but higher quality signal)
python scripts/train.py --problem_size 200 --use_lkh
```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--problem_size` | 200 | Number of customers |
| `--n_epochs` | 50 | Training epochs |
| `--steps_per_epoch` | 256 | Steps per epoch |
| `--batch_size` | 10 | Instances per step |
| `--width` | 10 | Samples per instance |
| `--depth` | 12 | EGNN layers |
| `--hidden_dim` | 128 | Hidden dimension |
| `--use_lkh` | False | Use LKH-3 for evaluation |

## Architecture

```
Input: CVRP Instance (coords, demands, capacity)
    ↓
Build E(2)-Equivariant Graph
    - Node features: (demand/capacity, distance_from_depot)
    - Edge features: (distance, affinity)
    - No polar angle θ!
    ↓
EGNN Layers (12 layers)
    - Message: invariant (uses distances)
    - Coordinate update: equivariant
    ↓
Edge → Heatmap H[i,j]
    ↓
Sequential Sampler (from GLOP)
    - Respects capacity constraints
    - Returns routes + log_probs
    ↓
Evaluate Routes (NN or LKH)
    ↓
REINFORCE Update
```

## Project Structure

```
edisco-partition/
├── edisco_partition/
│   ├── models/
│   │   ├── egnn.py           # E(2)-equivariant GNN
│   │   └── partition_net.py  # Full partition network
│   ├── data/
│   │   ├── instance.py       # CVRP generation
│   │   └── graph.py          # Graph construction
│   ├── sampler/
│   │   └── sequential.py     # Route sampler
│   └── evaluation/
│       ├── nn_routing.py     # Nearest neighbor
│       └── lkh_routing.py    # LKH-3
├── scripts/
│   └── train.py              # Training script
└── checkpoints/              # Saved models
```
