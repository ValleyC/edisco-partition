"""
Sequential Sampler for CVRP Route Construction.

Adapted from GLOP's sampler. Constructs routes by sequentially sampling
the next node based on:
1. Heatmap probabilities (learned)
2. Visit mask (don't revisit customers)
3. Capacity mask (respect vehicle capacity)
4. Depot mask (prevent unnecessary depot returns)
"""

import torch
from torch.distributions import Categorical
import math


class Sampler:
    """
    Sequential route sampler for CVRP.

    Given a heatmap H[i,j] = P(next=j | current=i), samples complete
    routes while respecting capacity constraints.
    """

    def __init__(self, demand, heatmap, capacity, batch_size, device):
        """
        Args:
            demand: (n_nodes,) - demands with depot=0 at index 0
            heatmap: (n_nodes, n_nodes) - transition probabilities
            capacity: Vehicle capacity
            batch_size: Number of routes to sample in parallel
            device: Device
        """
        self.n = demand.size(0)
        self.demand = demand.to(device)
        self.heatmap = heatmap.to(device)
        self.capacity = capacity
        self.max_vehicle = math.ceil(sum(self.demand) / capacity) + 1
        self.total_demand = self.demand.sum()
        self.bs = batch_size
        self.device = device

    def gen_subsets(self, require_prob=False, greedy_mode=False):
        """
        Generate complete routes.

        Args:
            require_prob: If True, return log probabilities for REINFORCE
            greedy_mode: If True, always pick highest probability (no sampling)

        Returns:
            routes: (batch_size, route_length) - sequences of node indices
            log_probs: (batch_size, route_length-1) - log probs (if require_prob)
        """
        if greedy_mode:
            assert not require_prob, "Cannot compute log_probs in greedy mode"

        # Initialize at depot
        actions = torch.zeros((self.bs,), dtype=torch.long, device=self.device)

        # Masks
        visit_mask = torch.ones(self.bs, self.n, device=self.device)
        visit_mask = self._update_visit_mask(visit_mask, actions)

        used_capacity = torch.zeros(self.bs, device=self.device)
        used_capacity, capacity_mask = self._update_capacity_mask(actions, used_capacity)

        vehicle_count = torch.zeros(self.bs, device=self.device)
        demand_count = torch.zeros(self.bs, device=self.device)
        depot_mask, vehicle_count, demand_count = self._update_depot_mask(
            vehicle_count, demand_count, actions, capacity_mask, visit_mask
        )

        paths_list = [actions]
        log_probs_list = []

        done = self._check_done(visit_mask, actions)

        while not done:
            actions, log_probs = self._pick_node(
                actions, visit_mask, capacity_mask, depot_mask,
                require_prob, greedy_mode
            )
            paths_list.append(actions)

            if require_prob:
                log_probs_list.append(log_probs)
                # Clone masks to avoid in-place modification issues
                visit_mask = visit_mask.clone()
                depot_mask = depot_mask.clone()

            visit_mask = self._update_visit_mask(visit_mask, actions)
            used_capacity, capacity_mask = self._update_capacity_mask(actions, used_capacity)
            depot_mask, vehicle_count, demand_count = self._update_depot_mask(
                vehicle_count, demand_count, actions, capacity_mask, visit_mask
            )
            done = self._check_done(visit_mask, actions)

        if require_prob:
            return (
                torch.stack(paths_list).permute(1, 0),
                torch.stack(log_probs_list).permute(1, 0)
            )
        else:
            return torch.stack(paths_list).permute(1, 0)

    def _pick_node(self, prev, visit_mask, capacity_mask, depot_mask, require_prob, greedy_mode):
        """Pick next node based on heatmap and masks."""
        # Get transition probabilities from current nodes
        heatmap_row = self.heatmap[prev]  # (bs, n_nodes)

        # Apply masks
        dist = heatmap_row * visit_mask * capacity_mask * depot_mask

        log_prob = None

        if not greedy_mode:
            try:
                categorical = Categorical(dist)
                item = categorical.sample()
                if require_prob:
                    log_prob = categorical.log_prob(item)
            except:
                # Fallback if Categorical fails (e.g., all zeros)
                dist = torch.softmax(torch.log(dist + 1e-10), dim=1)
                item = torch.multinomial(dist, num_samples=1).squeeze()
                if require_prob:
                    log_prob = torch.log(dist[torch.arange(self.bs, device=self.device), item] + 1e-10)
        else:
            _, item = dist.max(dim=1)

        return item, log_prob

    def _update_visit_mask(self, visit_mask, actions):
        """Mark visited nodes (except depot which can be revisited)."""
        visit_mask[torch.arange(self.bs, device=self.device), actions] = 0
        visit_mask[:, 0] = 1  # Depot can always be revisited

        # Exception: if just left depot and customers remain, can't return immediately
        just_left_depot = actions == 0
        customers_remain = (visit_mask[:, 1:] != 0).any(dim=1)
        visit_mask[just_left_depot & customers_remain, 0] = 0

        return visit_mask

    def _update_capacity_mask(self, cur_nodes, used_capacity):
        """Update capacity tracking and mask infeasible nodes."""
        capacity_mask = torch.ones(self.bs, self.n, device=self.device)

        # Reset capacity when returning to depot
        used_capacity[cur_nodes == 0] = 0

        # Add demand of current node
        used_capacity = used_capacity + self.demand[cur_nodes]

        # Mask nodes whose demand exceeds remaining capacity
        remaining_capacity = self.capacity - used_capacity
        remaining_capacity_expanded = remaining_capacity.unsqueeze(-1).expand(-1, self.n)
        demand_expanded = self.demand.unsqueeze(0).expand(self.bs, -1)
        capacity_mask[demand_expanded > remaining_capacity_expanded] = 0

        return used_capacity, capacity_mask

    def _update_depot_mask(self, vehicle_count, demand_count, actions, capacity_mask, visit_mask):
        """Control when depot can/must be visited."""
        depot_mask = torch.ones(self.bs, self.n, device=self.device)

        # Update counters
        vehicle_count[actions == 0] += 1
        demand_count += self.demand[actions]

        # Remaining demand
        remaining_demand = self.total_demand - demand_count

        # Don't allow depot if we might not have enough vehicles to serve remaining demand
        too_early_for_depot = remaining_demand > self.capacity * (self.max_vehicle - vehicle_count)
        depot_mask[too_early_for_depot, 0] = 0

        # Must return to depot if no feasible customers
        no_feasible_customers = ((visit_mask[:, 1:] * capacity_mask[:, 1:]) == 0).all(dim=1)
        depot_mask[no_feasible_customers, 0] = 1

        return depot_mask, vehicle_count, demand_count

    def _check_done(self, visit_mask, actions):
        """Check if all customers visited and all vehicles returned to depot."""
        all_customers_visited = (visit_mask[:, 1:] == 0).all()
        all_at_depot = (actions == 0).all()
        return all_customers_visited and all_at_depot
