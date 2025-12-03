"""
GLOP Training with EGNN - Direct copy of GLOP's training code.

Only changes from original GLOP:
1. Network: EGNN instead of EmbNet+ParNet
2. Node features: (demand, r) instead of (demand, r, theta) - removes theta for equivariance
3. Evaluation: Simple NN routing instead of revisers (no pretrained models needed)

Usage:
    python scripts/train_glop.py --problem_size 200 --n_epochs 50
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from dataclasses import dataclass
from typing import Optional


@dataclass
class GraphData:
    """Simple data class to replace torch_geometric.data.Data."""
    x: torch.Tensor  # Node features
    edge_index: torch.Tensor  # Edge indices
    edge_attr: torch.Tensor  # Edge features
    pos: torch.Tensor  # Node coordinates
import numpy as np
import math
import time
import argparse
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LR = 3e-4

K_SPARSE = {
    100: 50,
    200: 50,
    500: 100,
    1000: 100,
    2000: 200,
}

CAPACITIES = {
    100: 50.,
    200: 80.,
    500: 100.,
    1000: 200.,
    2000: 300.,
}


# ============================================================================
# DATA GENERATION (from GLOP, modified for equivariance)
# ============================================================================

def gen_inst(n, device):
    """Generate CVRP instance - same as GLOP."""
    capacity = CAPACITIES.get(n, 100.)
    coors = torch.rand(size=(n+1, 2), device=device)
    demand = torch.randint(1, 10, (n+1,), device=device).float()
    demand[0] = 0
    return coors, demand, capacity


def gen_distance_matrix(coordinates):
    """Compute pairwise distances."""
    return torch.norm(coordinates[:, None] - coordinates, dim=2, p=2)


def gen_pyg_data(coors, demand, capacity, k_sparse):
    """
    Generate PyG data - MODIFIED for equivariance.

    Changes from GLOP:
    - Node features: (demand/cap, r) instead of (demand/cap, r, theta)
    - theta is removed because it breaks rotation equivariance
    - Edge construction: distance-based k-NN (rotation invariant)
    """
    n_nodes = demand.size(0)
    device = coors.device

    # Node features (EQUIVARIANT - no theta!)
    norm_demand = demand / capacity
    shift_coors = coors - coors[0]  # Shift to depot
    r = torch.norm(shift_coors, dim=1)  # Distance from depot (rotation invariant)

    # 2D features instead of 3D (no theta)
    x = torch.stack((norm_demand, r), dim=1)  # (n_nodes, 2)

    # Edge construction using distance (rotation invariant)
    dist_mat = gen_distance_matrix(coors)

    # k-NN based on distance (smaller = closer = higher priority)
    affinity_mat = -dist_mat.clone()
    affinity_mat.fill_diagonal_(float('-inf'))

    topk_values, topk_indices = torch.topk(affinity_mat, k=k_sparse, dim=1, largest=True)

    edge_index = torch.stack([
        torch.repeat_interleave(torch.arange(n_nodes, device=device), repeats=k_sparse),
        topk_indices.flatten()
    ])

    # Edge features: distance and affinity (both rotation invariant)
    edge_dist = dist_mat[edge_index[0], edge_index[1]].unsqueeze(1)
    max_dist = dist_mat.max() + 1e-8
    edge_affinity = (1 - edge_dist / max_dist)
    edge_attr = torch.cat([edge_dist, edge_affinity], dim=1)

    # Include coordinates for EGNN
    pyg_data = GraphData(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=coors)
    return pyg_data


# ============================================================================
# SAMPLER (exact copy from GLOP)
# ============================================================================

class Sampler:
    """Sequential sampler - EXACT COPY from GLOP."""

    def __init__(self, demand, heatmap, capacity, bs, device):
        self.n = demand.size(0)
        self.demand = demand.to(device)
        self.heatmap = heatmap.to(device)
        self.capacity = capacity
        self.max_vehicle = math.ceil(sum(self.demand).item() / capacity) + 1
        self.total_demand = self.demand.sum()
        self.bs = bs
        self.device = device

    def gen_subsets(self, require_prob=False, greedy_mode=False):
        if greedy_mode:
            assert not require_prob

        actions = torch.zeros((self.bs,), dtype=torch.long, device=self.device)
        visit_mask = torch.ones(size=(self.bs, self.n), device=self.device)
        visit_mask = self.update_visit_mask(visit_mask, actions)
        used_capacity = torch.zeros(size=(self.bs,), device=self.device)
        used_capacity, capacity_mask = self.update_capacity_mask(actions, used_capacity)

        vehicle_count = torch.zeros((self.bs,), device=self.device)
        demand_count = torch.zeros((self.bs,), device=self.device)
        depot_mask, vehicle_count, demand_count = self.update_depot_mask(
            vehicle_count, demand_count, actions, capacity_mask, visit_mask
        )

        paths_list = [actions]
        log_probs_list = []
        done = self.check_done(visit_mask, actions)

        while not done:
            actions, log_probs = self.pick_node(
                actions, visit_mask, capacity_mask, depot_mask, require_prob, greedy_mode
            )
            paths_list.append(actions)
            if require_prob:
                log_probs_list.append(log_probs)
                visit_mask = visit_mask.clone()
                depot_mask = depot_mask.clone()
            visit_mask = self.update_visit_mask(visit_mask, actions)
            used_capacity, capacity_mask = self.update_capacity_mask(actions, used_capacity)
            depot_mask, vehicle_count, demand_count = self.update_depot_mask(
                vehicle_count, demand_count, actions, capacity_mask, visit_mask
            )
            done = self.check_done(visit_mask, actions)

        if require_prob:
            return torch.stack(paths_list).permute(1, 0), torch.stack(log_probs_list).permute(1, 0)
        else:
            return torch.stack(paths_list).permute(1, 0)

    def pick_node(self, prev, visit_mask, capacity_mask, depot_mask, require_prob, greedy_mode=False):
        log_prob = None
        heatmap = self.heatmap[prev]
        dist = heatmap * visit_mask * capacity_mask * depot_mask

        if not greedy_mode:
            try:
                dist_cat = Categorical(dist)
                item = dist_cat.sample()
                log_prob = dist_cat.log_prob(item) if require_prob else None
            except:
                dist = torch.softmax(torch.log(dist + 1e-10), dim=1)
                item = torch.multinomial(dist, num_samples=1).squeeze()
                log_prob = torch.log(dist[torch.arange(self.bs, device=self.device), item] + 1e-10)
        else:
            _, item = dist.max(dim=1)
        return item, log_prob

    def update_depot_mask(self, vehicle_count, demand_count, actions, capacity_mask, visit_mask):
        depot_mask = torch.ones((self.bs, self.n), device=self.device)
        vehicle_count[actions == 0] += 1
        demand_count += self.demand[actions]
        remaining_demand = self.total_demand - demand_count
        depot_mask[remaining_demand > self.capacity * (self.max_vehicle - vehicle_count), 0] = 0
        depot_mask[((visit_mask[:, 1:] * capacity_mask[:, 1:]) == 0).all(dim=1), 0] = 1
        return depot_mask, vehicle_count, demand_count

    def update_visit_mask(self, visit_mask, actions):
        visit_mask[torch.arange(self.bs, device=self.device), actions] = 0
        visit_mask[:, 0] = 1
        visit_mask[(actions == 0) * (visit_mask[:, 1:] != 0).any(dim=1), 0] = 0
        return visit_mask

    def update_capacity_mask(self, cur_nodes, used_capacity):
        capacity_mask = torch.ones(size=(self.bs, self.n), device=self.device)
        used_capacity[cur_nodes == 0] = 0
        used_capacity = used_capacity + self.demand[cur_nodes]
        remaining_capacity = self.capacity - used_capacity
        remaining_capacity_repeat = remaining_capacity.unsqueeze(-1).repeat(1, self.n)
        demand_repeat = self.demand.unsqueeze(0).repeat(self.bs, 1)
        capacity_mask[demand_repeat > remaining_capacity_repeat] = 0
        return used_capacity, capacity_mask

    def check_done(self, visit_mask, actions):
        return (visit_mask[:, 1:] == 0).all() and (actions == 0).all()


# ============================================================================
# SIMPLE EVALUATION (replaces GLOP's revisers)
# ============================================================================

def eval_routes(coors, routes):
    """
    Evaluate routes using nearest neighbor TSP.
    Replaces GLOP's reviser-based evaluation.
    """
    bs = routes.size(0)
    costs = []
    coors_np = coors.cpu().numpy()
    depot = coors_np[0]

    for i in range(bs):
        route = routes[i].cpu().numpy()
        cost = 0.0

        # Split route at depot visits
        segments = []
        current_seg = []
        for node in route:
            if node == 0:
                if current_seg:
                    segments.append(current_seg)
                    current_seg = []
            else:
                current_seg.append(node)
        if current_seg:
            segments.append(current_seg)

        # Compute distance for each segment
        for seg in segments:
            if not seg:
                continue
            seg_coords = coors_np[seg]
            n = len(seg)

            if n == 1:
                cost += 2 * np.linalg.norm(depot - seg_coords[0])
                continue

            # Simple nearest neighbor
            visited = [False] * n
            current = depot
            for _ in range(n):
                best_d, best_j = float('inf'), -1
                for j in range(n):
                    if not visited[j]:
                        d = np.linalg.norm(current - seg_coords[j])
                        if d < best_d:
                            best_d, best_j = d, j
                if best_j >= 0:
                    visited[best_j] = True
                    cost += best_d
                    current = seg_coords[best_j]
            cost += np.linalg.norm(current - depot)

        costs.append(cost)

    return torch.tensor(costs, device=coors.device)


# ============================================================================
# LKH EVALUATION (optional, for higher quality training signal)
# ============================================================================

import subprocess
import tempfile
import shutil

def find_lkh():
    """Find LKH executable."""
    # Check environment variable
    env_path = os.environ.get('LKH_PATH')
    if env_path and os.path.isfile(env_path):
        return env_path
    # Check system PATH
    system_lkh = shutil.which('LKH')
    if system_lkh:
        return system_lkh
    # Common paths
    home = os.path.expanduser("~")
    for path in [f"{home}/LKH_install/LKH-3.0.8/LKH", f"{home}/LKH_install/LKH-3.0.6/LKH",
                 f"{home}/Code/CVRP/LKH-3.0.8/LKH", f"{home}/LKH-3.0.8/LKH", f"{home}/LKH-3/LKH",
                 "/usr/local/bin/LKH", "./LKH-3.0.8/LKH", "./LKH"]:
        if os.path.isfile(path):
            return path
    return None

LKH_PATH = find_lkh()

def eval_routes_lkh(coors, routes, capacity, demand, time_limit=1, max_trials=100):
    """
    Evaluate routes using LKH-3 for each segment (TSP).
    Much higher quality than NN, but slower.

    Args:
        time_limit: seconds per segment (default 1)
        max_trials: LKH trials per segment (default 100)
    """
    if LKH_PATH is None:
        return eval_routes(coors, routes)  # Fallback to NN

    bs = routes.size(0)
    costs = []
    coors_np = coors.cpu().numpy()
    depot = coors_np[0]

    for i in range(bs):
        route = routes[i].cpu().numpy()
        cost = 0.0

        # Split route at depot visits
        segments = []
        current_seg = []
        for node in route:
            if node == 0:
                if current_seg:
                    segments.append(current_seg)
                    current_seg = []
            else:
                current_seg.append(node)
        if current_seg:
            segments.append(current_seg)

        # Solve TSP for each segment with LKH
        for seg in segments:
            if not seg:
                continue
            n = len(seg)
            if n == 1:
                cost += 2 * np.linalg.norm(depot - coors_np[seg[0]])
                continue
            if n == 2:
                cost += np.linalg.norm(depot - coors_np[seg[0]])
                cost += np.linalg.norm(coors_np[seg[0]] - coors_np[seg[1]])
                cost += np.linalg.norm(coors_np[seg[1]] - depot)
                continue

            # Build TSP with depot
            seg_with_depot = [0] + seg
            seg_coords = coors_np[seg_with_depot]

            try:
                seg_cost = solve_tsp_lkh(seg_coords, time_limit, max_trials)
                cost += seg_cost
            except:
                # Fallback to NN for this segment
                seg_coords_only = coors_np[seg]
                visited = [False] * n
                current = depot
                seg_cost = 0.0
                for _ in range(n):
                    best_d, best_j = float('inf'), -1
                    for j in range(n):
                        if not visited[j]:
                            d = np.linalg.norm(current - seg_coords_only[j])
                            if d < best_d:
                                best_d, best_j = d, j
                    if best_j >= 0:
                        visited[best_j] = True
                        seg_cost += best_d
                        current = seg_coords_only[best_j]
                seg_cost += np.linalg.norm(current - depot)
                cost += seg_cost

        costs.append(cost)

    return torch.tensor(costs, device=coors.device)


def solve_tsp_lkh(coords, time_limit=1, max_trials=100):
    """Solve TSP using LKH-3."""
    n = len(coords)
    scale = 100000

    with tempfile.TemporaryDirectory() as tmpdir:
        prob_file = os.path.join(tmpdir, "problem.tsp")
        par_file = os.path.join(tmpdir, "param.par")
        tour_file = os.path.join(tmpdir, "tour.txt")

        # Write TSP file
        with open(prob_file, 'w') as f:
            f.write(f"NAME : seg\nTYPE : TSP\nDIMENSION : {n}\n")
            f.write("EDGE_WEIGHT_TYPE : EUC_2D\nNODE_COORD_SECTION\n")
            for i, (x, y) in enumerate(coords):
                f.write(f"{i+1} {int(x*scale)} {int(y*scale)}\n")
            f.write("EOF\n")

        # Write param file
        with open(par_file, 'w') as f:
            f.write(f"PROBLEM_FILE = {prob_file}\n")
            f.write(f"OUTPUT_TOUR_FILE = {tour_file}\n")
            f.write(f"TIME_LIMIT = {time_limit}\n")
            f.write(f"MAX_TRIALS = {max_trials}\n")
            f.write("RUNS = 1\n")
            f.write("SEED = 1\n")

        # Run LKH
        subprocess.run([LKH_PATH, par_file], capture_output=True, timeout=time_limit+5)

        # Parse tour and compute distance
        with open(tour_file, 'r') as f:
            lines = f.readlines()

        tour = []
        in_tour = False
        for line in lines:
            if "TOUR_SECTION" in line:
                in_tour = True
                continue
            if in_tour:
                node = int(line.strip())
                if node == -1:
                    break
                tour.append(node - 1)  # 0-indexed

        # Compute tour length
        cost = 0.0
        for i in range(len(tour)):
            cost += np.linalg.norm(coords[tour[i]] - coords[tour[(i+1) % len(tour)]])

        return cost


# ============================================================================
# EGNN NETWORK (replaces GLOP's EmbNet+ParNet)
# ============================================================================

class EGNNLayer(nn.Module):
    """E(n)-Equivariant GNN Layer."""

    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Message MLP
        self.msg_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + 1 + edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # Node update
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, node_dim),
        )

        # Edge update
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, edge_dim),
        )

        # Coordinate update
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1, bias=False),
        )
        nn.init.zeros_(self.coord_mlp[-1].weight)

        self.node_norm = nn.LayerNorm(node_dim)
        self.edge_norm = nn.LayerNorm(edge_dim)

    def forward(self, h, x, edge_index, e):
        src, dst = edge_index
        n_nodes = h.size(0)

        # Compute distances (invariant)
        rel_vec = x[dst] - x[src]
        dist = rel_vec.norm(dim=-1, keepdim=True)

        # Messages
        msg_in = torch.cat([h[src], h[dst], dist, e], dim=-1)
        msg = self.msg_mlp(msg_in)

        # Aggregate
        agg = torch.zeros(n_nodes, self.hidden_dim, device=h.device)
        agg.index_add_(0, dst, msg)

        # Update nodes
        h_new = self.node_norm(h + self.node_mlp(torch.cat([h, agg], dim=-1)))

        # Update edges
        e_new = self.edge_norm(e + self.edge_mlp(torch.cat([e, msg], dim=-1)))

        # Update coordinates (equivariant)
        coord_w = torch.tanh(self.coord_mlp(msg))
        rel_dir = rel_vec / (dist + 1e-8)
        coord_delta = torch.zeros_like(x)
        coord_delta.index_add_(0, dst, coord_w * rel_dir)
        x_new = x + 0.1 * coord_delta

        return h_new, x_new, e_new


class EGNNNet(nn.Module):
    """EGNN-based partition network - replaces GLOP's Net."""

    def __init__(self, units, feats, k_sparse, edge_feats=2, depth=12):
        super().__init__()
        self.k_sparse = k_sparse

        # Embeddings
        self.node_embed = nn.Sequential(
            nn.Linear(feats, units),
            nn.SiLU(),
            nn.Linear(units, units),
        )
        self.edge_embed = nn.Sequential(
            nn.Linear(edge_feats, units),
            nn.SiLU(),
            nn.Linear(units, units),
        )

        # EGNN layers
        self.layers = nn.ModuleList([
            EGNNLayer(units, units, units * 2) for _ in range(depth)
        ])

        # Output head
        self.out_head = nn.Sequential(
            nn.Linear(units, units),
            nn.SiLU(),
            nn.Linear(units, 1),
        )
        nn.init.xavier_uniform_(self.out_head[-1].weight, gain=0.1)
        nn.init.zeros_(self.out_head[-1].bias)

    def forward(self, pyg):
        h = self.node_embed(pyg.x)
        e = self.edge_embed(pyg.edge_attr)
        x = pyg.pos.clone()

        for layer in self.layers:
            h, x, e = layer(h, x, pyg.edge_index, e)

        # Edge logits -> softmax per source
        logits = self.out_head(e).squeeze(-1)
        logits = logits.view(-1, self.k_sparse)
        probs = F.softmax(logits, dim=-1)
        return probs.flatten()

    def reshape(self, pyg, vector):
        """Convert flat vector to heatmap matrix."""
        n_nodes = pyg.x.size(0)
        matrix = torch.zeros(n_nodes, n_nodes, device=vector.device)
        matrix[pyg.edge_index[0], pyg.edge_index[1]] = vector
        return matrix


# ============================================================================
# TRAINING (exact structure from GLOP)
# ============================================================================

def infer_heatmap(model, pyg_data):
    """Same as GLOP."""
    heatmap = model(pyg_data)
    heatmap = heatmap / (heatmap.min() + 1e-5)
    heatmap = model.reshape(pyg_data, heatmap) + 1e-5
    return heatmap


def train_batch(model, optimizer, n, bs, opts):
    """Training batch - same structure as GLOP."""
    model.train()
    loss_lst = []

    for _ in range(opts.batch_size):
        coors, demand, capacity = gen_inst(n, DEVICE)
        pyg_data = gen_pyg_data(coors, demand, capacity, K_SPARSE.get(n, 50))
        heatmap = infer_heatmap(model, pyg_data)

        sampler = Sampler(demand, heatmap, capacity, bs, DEVICE)
        routes, log_probs = sampler.gen_subsets(require_prob=True)

        # Evaluation - use LKH if enabled, otherwise NN
        if opts.use_lkh and LKH_PATH is not None:
            objs = eval_routes_lkh(coors, routes, capacity, demand,
                                   time_limit=opts.lkh_time_limit,
                                   max_trials=opts.lkh_max_trials)
        else:
            objs = eval_routes(coors, routes)

        # REINFORCE - exact same as GLOP
        baseline = objs.mean()
        log_probs = log_probs.to(DEVICE)
        reinforce_loss = torch.sum((objs - baseline) * log_probs.sum(dim=1)) / bs
        loss_lst.append(reinforce_loss)

    loss = sum(loss_lst) / opts.batch_size
    optimizer.zero_grad()
    loss.backward()

    if not opts.no_clip:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=opts.max_norm, norm_type=2)

    optimizer.step()
    return loss.item(), objs.mean().item()


def train_epoch(n, bs, steps_per_epoch, net, optimizer, scheduler, opts):
    """Same as GLOP."""
    losses, objs = [], []
    for _ in tqdm(range(steps_per_epoch), desc=f"Training"):
        loss, obj = train_batch(net, optimizer, n, bs, opts)
        losses.append(loss)
        objs.append(obj)
    scheduler.step()
    return np.mean(losses), np.mean(objs)


@torch.no_grad()
def validation(n, net, opts):
    """Same as GLOP."""
    net.eval()
    sum_obj = 0
    for _ in range(opts.val_size):
        coors, demand, capacity = gen_inst(n, DEVICE)
        pyg_data = gen_pyg_data(coors, demand, capacity, K_SPARSE.get(n, 50))
        heatmap = infer_heatmap(net, pyg_data)
        sampler = Sampler(demand, heatmap, capacity, 1, DEVICE)
        routes = sampler.gen_subsets(require_prob=False, greedy_mode=True)
        # Use LKH for validation if enabled
        if opts.use_lkh and LKH_PATH is not None:
            obj = eval_routes_lkh(coors, routes, capacity, demand,
                                  time_limit=opts.lkh_time_limit,
                                  max_trials=opts.lkh_max_trials).min()
        else:
            obj = eval_routes(coors, routes).min()
        sum_obj += obj.item()
    return sum_obj / opts.val_size


@torch.no_grad()
def check_equivariance(net, n, device):
    """Verify rotation equivariance."""
    net.eval()
    coors, demand, capacity = gen_inst(min(n, 100), device)
    pyg1 = gen_pyg_data(coors, demand, capacity, K_SPARSE.get(n, 50))
    h1 = infer_heatmap(net, pyg1)

    # Rotate 90 degrees
    theta = torch.tensor(np.pi / 2, device=device)
    R = torch.tensor([[torch.cos(theta), -torch.sin(theta)],
                      [torch.sin(theta), torch.cos(theta)]], device=device)
    coors_rot = coors @ R.T
    pyg2 = gen_pyg_data(coors_rot, demand, capacity, K_SPARSE.get(n, 50))
    h2 = infer_heatmap(net, pyg2)

    return (h1 - h2).abs().max().item()


def train(n, bs, steps_per_epoch, n_epochs, opts):
    """Main training - same structure as GLOP."""
    k_sparse = K_SPARSE.get(n, 50)

    # Create EGNN network (instead of GLOP's Net)
    net = EGNNNet(
        units=opts.units,
        feats=2,  # (demand, r) - no theta!
        k_sparse=k_sparse,
        edge_feats=2,
        depth=opts.depth
    ).to(DEVICE)

    print(f"Parameters: {sum(p.numel() for p in net.parameters()):,}")

    optimizer = torch.optim.AdamW(net.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    if opts.checkpoint_path:
        checkpoint = torch.load(opts.checkpoint_path, map_location=DEVICE)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        starting_epoch = checkpoint['epoch'] + 1
    else:
        starting_epoch = 1

    # Check equivariance
    diff = check_equivariance(net, n, DEVICE)
    print(f"Equivariance check: max_diff={diff:.6f} [{'PASS' if diff < 0.01 else 'FAIL'}]")

    # Initial validation
    best_avg_obj = validation(n, net, opts)
    print(f'epoch 0: {best_avg_obj:.4f}')

    # Training loop
    sum_time = 0
    for epoch in range(starting_epoch, n_epochs + 1):
        start = time.time()
        avg_loss, train_obj = train_epoch(n, bs, steps_per_epoch, net, optimizer, scheduler, opts)
        sum_time += time.time() - start

        avg_obj = validation(n, net, opts)
        print(f'epoch {epoch}: val={avg_obj:.4f}, train={train_obj:.4f}, loss={avg_loss:.4f}')

        if best_avg_obj > avg_obj:
            best_avg_obj = avg_obj
            print(f'  -> New best! Saving...')
            os.makedirs('checkpoints', exist_ok=True)
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }
            torch.save(checkpoint, f'checkpoints/egnn-cvrp{n}-best.pt')

        if epoch % 5 == 0:
            diff = check_equivariance(net, n, DEVICE)
            print(f'  Equivariance: {diff:.6f}')

    print(f'Total training time: {sum_time:.1f}s')
    print(f'Best validation: {best_avg_obj:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem_size', type=int, default=200)
    parser.add_argument('--width', type=int, default=10, help='Samples per instance')
    parser.add_argument('--val_size', type=int, default=100)
    parser.add_argument('--n_epochs', type=int, default=50)
    parser.add_argument('--steps_per_epoch', type=int, default=256)
    parser.add_argument('--checkpoint_path', type=str, default='')
    parser.add_argument('--max_norm', type=float, default=1.0)
    parser.add_argument('--units', type=int, default=48)
    parser.add_argument('--no_clip', action='store_true')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--depth', type=int, default=12)
    parser.add_argument('--seed', type=int, default=1)
    # LKH options
    parser.add_argument('--use_lkh', action='store_true', help='Use LKH-3 for route evaluation')
    parser.add_argument('--lkh_time_limit', type=int, default=1, help='LKH time limit per segment (seconds)')
    parser.add_argument('--lkh_max_trials', type=int, default=100, help='LKH max trials per segment')

    opts = parser.parse_args()

    torch.manual_seed(opts.seed)
    np.random.seed(opts.seed)

    print("=" * 60)
    print("GLOP Training with EGNN (Equivariant)")
    print("=" * 60)
    print(f"Problem size: {opts.problem_size}")
    print(f"Network: EGNN {opts.depth} layers, {opts.units} units")
    print(f"Width (samples): {opts.width}")
    print(f"Batch size: {opts.batch_size}")
    print(f"Device: {DEVICE}")
    if opts.use_lkh:
        if LKH_PATH:
            print(f"LKH: enabled (time={opts.lkh_time_limit}s, trials={opts.lkh_max_trials})")
        else:
            print("LKH: requested but NOT FOUND - falling back to NN")
    else:
        print("LKH: disabled (using NN routing)")
    print("=" * 60)

    train(opts.problem_size, opts.width, opts.steps_per_epoch, opts.n_epochs, opts)
