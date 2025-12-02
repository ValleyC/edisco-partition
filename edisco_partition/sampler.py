"""
Sequential Sampler for CVRP Route Construction.

This module implements GLOP-style sequential sampling with constraint handling.
The sampler constructs routes step-by-step using a heatmap (transition probability matrix)
while respecting capacity and visitation constraints.

Key features:
- Sequential construction ensures feasibility by design
- Low-variance gradient estimation for REINFORCE
- Constraint masks handle capacity and depot returns automatically
"""
import torch
from torch.distributions import Categorical
import math
from typing import Tuple, Optional


class SequentialSampler:
    """
    Sequential route sampler with constraint handling.

    Given a heatmap H where H[i,j] is the probability of transitioning from
    node i to node j, this sampler constructs feasible CVRP routes by:
    1. Starting at depot (node 0)
    2. Sequentially selecting next nodes based on heatmap probabilities
    3. Applying masks to ensure capacity and visitation constraints
    4. Returning to depot when vehicle is full

    This approach has much lower variance than independent sampling because:
    - Each step conditions on previous decisions
    - Constraints are built into the sampling process
    - The action space is O(n) per step, not O(k^n) joint space
    """

    def __init__(
        self,
        demand: torch.Tensor,
        heatmap: torch.Tensor,
        capacity: float,
        n_samples: int,
        device: torch.device
    ):
        """
        Initialize sampler.

        Args:
            demand: (n_nodes,) demand for each node (depot=0)
            heatmap: (n_nodes, n_nodes) transition probability matrix
            capacity: Vehicle capacity
            n_samples: Number of parallel samples (batch dimension)
            device: Torch device
        """
        self.n_nodes = demand.size(0)
        self.demand = demand.to(device)
        self.heatmap = heatmap.to(device)
        self.capacity = capacity
        self.n_samples = n_samples
        self.device = device

        # Compute max vehicles needed
        self.total_demand = self.demand.sum()
        self.max_vehicles = math.ceil(self.total_demand.item() / capacity) + 1

    def sample(
        self,
        require_log_prob: bool = False,
        greedy: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Sample routes using sequential construction.

        Args:
            require_log_prob: Whether to return log probabilities
            greedy: If True, use argmax instead of sampling

        Returns:
            routes: (n_samples, seq_len) tensor of node indices
            log_probs: (n_samples, seq_len-1) log probabilities (if require_log_prob)
        """
        if greedy:
            assert not require_log_prob, "Cannot compute log_prob in greedy mode"

        # Initialize state
        current_node = torch.zeros(self.n_samples, dtype=torch.long, device=self.device)

        # Masks
        visit_mask = torch.ones(self.n_samples, self.n_nodes, device=self.device)
        visit_mask = self._update_visit_mask(visit_mask, current_node)

        used_capacity = torch.zeros(self.n_samples, device=self.device)
        used_capacity, capacity_mask = self._update_capacity_mask(current_node, used_capacity)

        vehicle_count = torch.zeros(self.n_samples, device=self.device)
        demand_served = torch.zeros(self.n_samples, device=self.device)
        depot_mask = self._update_depot_mask(
            vehicle_count, demand_served, current_node, capacity_mask, visit_mask
        )

        # Collect route and log probs
        route_nodes = [current_node]
        log_probs_list = []

        # Sequential construction
        done = self._check_done(visit_mask, current_node)

        while not done:
            # Select next node
            next_node, log_prob = self._select_next(
                current_node, visit_mask, capacity_mask, depot_mask,
                require_log_prob, greedy
            )

            route_nodes.append(next_node)
            if require_log_prob:
                log_probs_list.append(log_prob)

            # Update state
            if require_log_prob:
                # Need to clone for gradient computation
                visit_mask = visit_mask.clone()
                depot_mask = depot_mask.clone()

            visit_mask = self._update_visit_mask(visit_mask, next_node)
            used_capacity, capacity_mask = self._update_capacity_mask(next_node, used_capacity)

            # Update vehicle count when returning to depot
            vehicle_count = vehicle_count + (next_node == 0).float()
            demand_served = demand_served + self.demand[next_node]

            depot_mask = self._update_depot_mask(
                vehicle_count, demand_served, next_node, capacity_mask, visit_mask
            )

            current_node = next_node
            done = self._check_done(visit_mask, current_node)

        # Stack results
        routes = torch.stack(route_nodes, dim=1)  # (n_samples, seq_len)

        if require_log_prob:
            log_probs = torch.stack(log_probs_list, dim=1)  # (n_samples, seq_len-1)
            return routes, log_probs
        else:
            return routes, None

    def _select_next(
        self,
        current: torch.Tensor,
        visit_mask: torch.Tensor,
        capacity_mask: torch.Tensor,
        depot_mask: torch.Tensor,
        require_log_prob: bool,
        greedy: bool
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Select next node based on heatmap and masks."""
        # Get transition probabilities from current node
        probs = self.heatmap[current]  # (n_samples, n_nodes)

        # Apply masks
        masked_probs = probs * visit_mask * capacity_mask * depot_mask

        # Normalize (add small epsilon for numerical stability)
        masked_probs = masked_probs + 1e-10
        masked_probs = masked_probs / masked_probs.sum(dim=1, keepdim=True)

        if greedy:
            next_node = masked_probs.argmax(dim=1)
            return next_node, None
        else:
            try:
                dist = Categorical(masked_probs)
                next_node = dist.sample()
                log_prob = dist.log_prob(next_node) if require_log_prob else None
            except Exception:
                # Fallback for numerical issues
                next_node = torch.multinomial(masked_probs, num_samples=1).squeeze(1)
                if require_log_prob:
                    log_prob = torch.log(masked_probs[torch.arange(self.n_samples), next_node] + 1e-10)
                else:
                    log_prob = None

            return next_node, log_prob

    def _update_visit_mask(
        self,
        visit_mask: torch.Tensor,
        current: torch.Tensor
    ) -> torch.Tensor:
        """Update visitation mask after visiting a node."""
        # Mark current node as visited
        visit_mask[torch.arange(self.n_samples, device=self.device), current] = 0

        # Depot can always be revisited
        visit_mask[:, 0] = 1

        # Exception: can't stay at depot if there are unvisited customers
        at_depot = (current == 0)
        has_unvisited = (visit_mask[:, 1:] != 0).any(dim=1)
        visit_mask[at_depot & has_unvisited, 0] = 0

        return visit_mask

    def _update_capacity_mask(
        self,
        current: torch.Tensor,
        used_capacity: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update capacity constraint mask."""
        capacity_mask = torch.ones(self.n_samples, self.n_nodes, device=self.device)

        # Reset capacity at depot
        used_capacity = used_capacity.clone()
        used_capacity[current == 0] = 0

        # Add current node's demand
        used_capacity = used_capacity + self.demand[current]

        # Mask nodes that would exceed capacity
        remaining = self.capacity - used_capacity  # (n_samples,)
        demand_broadcast = self.demand.unsqueeze(0).expand(self.n_samples, -1)
        capacity_mask[demand_broadcast > remaining.unsqueeze(1)] = 0

        return used_capacity, capacity_mask

    def _update_depot_mask(
        self,
        vehicle_count: torch.Tensor,
        demand_served: torch.Tensor,
        current: torch.Tensor,
        capacity_mask: torch.Tensor,
        visit_mask: torch.Tensor
    ) -> torch.Tensor:
        """Update depot mask to control vehicle usage."""
        depot_mask = torch.ones(self.n_samples, self.n_nodes, device=self.device)

        # Remaining demand to serve
        remaining_demand = self.total_demand - demand_served

        # Can't return to depot if remaining demand needs more vehicles than available
        remaining_vehicles = self.max_vehicles - vehicle_count
        depot_mask[remaining_demand > self.capacity * remaining_vehicles, 0] = 0

        # Must return to depot if no valid customers left
        valid_customers = visit_mask[:, 1:] * capacity_mask[:, 1:]
        no_valid_customers = (valid_customers == 0).all(dim=1)
        depot_mask[no_valid_customers, 0] = 1

        return depot_mask

    def _check_done(
        self,
        visit_mask: torch.Tensor,
        current: torch.Tensor
    ) -> bool:
        """Check if all routes are complete."""
        all_visited = (visit_mask[:, 1:] == 0).all()
        all_at_depot = (current == 0).all()
        return all_visited and all_at_depot


def routes_to_segments(routes: torch.Tensor) -> list:
    """
    Convert flat route tensor to list of route segments.

    Args:
        routes: (seq_len,) tensor of node indices

    Returns:
        List of route segments, each a list of customer indices
    """
    routes_np = routes.cpu().numpy()
    segments = []
    current_segment = []

    for node in routes_np:
        if node == 0:
            if current_segment:
                segments.append(current_segment)
                current_segment = []
        else:
            current_segment.append(int(node))

    if current_segment:
        segments.append(current_segment)

    return segments


def compute_route_distance(
    coords: torch.Tensor,
    routes: torch.Tensor
) -> torch.Tensor:
    """
    Compute total route distance for sampled routes.

    Args:
        coords: (n_nodes, 2) coordinates
        routes: (n_samples, seq_len) route sequences

    Returns:
        distances: (n_samples,) total distance per route
    """
    n_samples, seq_len = routes.shape
    device = coords.device

    distances = torch.zeros(n_samples, device=device)

    for i in range(n_samples):
        route = routes[i]
        dist = 0.0

        for t in range(seq_len - 1):
            from_node = route[t].item()
            to_node = route[t + 1].item()
            dist += torch.norm(coords[to_node] - coords[from_node]).item()

        distances[i] = dist

    return distances
