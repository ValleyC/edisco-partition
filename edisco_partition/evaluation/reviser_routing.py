"""
Reviser-based routing evaluation for CVRP partition training.

Uses GLOP's pretrained Reviser models to evaluate sub-routes during training.
This provides faster GPU-accelerated evaluation compared to LKH.
"""
import os
import torch
import numpy as np
from typing import List, Optional
from argparse import Namespace

# Import GLOP modules
from ..glop.utils.functions import load_model, reconnect
from ..glop.utils.insertion import random_insertion_parallel, HAS_RANDOM_INSERTION
from ..glop.problems import TSP


def get_pretrained_path():
    """Get path to pretrained Reviser models."""
    # Look in common locations
    module_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    candidates = [
        os.path.join(module_dir, 'pretrained'),
        os.path.join(module_dir, '..', 'pretrained'),
        'pretrained',
    ]
    for path in candidates:
        if os.path.exists(path):
            return os.path.abspath(path)
    return None


class ReviserEvaluator:
    """
    Evaluates TSP sub-routes using GLOP's pretrained Reviser models.

    This class follows GLOP's exact evaluation approach:
    1. Convert partition routes to TSP instances
    2. Initialize with random insertion heuristic
    3. Improve with Reviser neural network
    4. Sum costs per route

    Usage:
        evaluator = ReviserEvaluator(revision_lens=[20], device='cuda')
        costs = evaluator.eval_routes(coords, routes)
    """

    def __init__(
        self,
        revision_lens: List[int] = [20],
        revision_iters: List[int] = [5],
        pretrained_path: Optional[str] = None,
        device: str = 'cuda',
        no_aug: bool = False,
        decode_strategy: str = 'greedy',
    ):
        """
        Initialize Reviser evaluator.

        Args:
            revision_lens: List of reviser sizes to use (e.g., [20])
            revision_iters: Number of revision iterations per reviser
            pretrained_path: Path to pretrained models (auto-detected if None)
            device: Device to run on ('cuda' or 'cpu')
            no_aug: Disable augmentation (faster but slightly worse)
            decode_strategy: 'greedy' or 'sampling'
        """
        self.device = device
        self.revision_lens = revision_lens
        self.revision_iters = revision_iters
        self.no_aug = no_aug

        # Find pretrained path
        if pretrained_path is None:
            pretrained_path = get_pretrained_path()
        if pretrained_path is None:
            raise FileNotFoundError(
                "Could not find pretrained Reviser models. "
                "Please specify pretrained_path or ensure 'pretrained/' directory exists."
            )
        self.pretrained_path = pretrained_path

        # Load revisers
        self.revisers = []
        for reviser_size in revision_lens:
            reviser_path = os.path.join(
                pretrained_path,
                f'Reviser-stage2/reviser_{reviser_size}/epoch-299.pt'
            )
            if not os.path.exists(reviser_path):
                raise FileNotFoundError(f"Reviser model not found: {reviser_path}")

            reviser, _ = load_model(reviser_path, is_local=True)
            reviser.to(device)
            reviser.eval()
            reviser.set_decode_type(decode_strategy)
            self.revisers.append(reviser)

        # TSP cost function (returns SHPP cost when return_local=True)
        self.get_cost_func = lambda input, pi: TSP.get_costs(input, pi, return_local=True)

        # Options namespace for reconnect
        self.opts = Namespace(
            revision_lens=revision_lens,
            revision_iters=revision_iters,
            no_aug=no_aug,
            no_prune=True,  # Don't prune during training eval
            eval_batch_size=1,
        )

    def _trans_tsp(self, coors: torch.Tensor, routes: torch.Tensor, min_reviser_size: int = 20):
        """
        Convert CVRP routes to TSP instances for Reviser evaluation.

        Following GLOP's trans_tsp function exactly.

        Args:
            coors: Coordinates (n_nodes+1, 2) including depot at index 0
            routes: Route tensor (width, max_route_len) with node indices

        Returns:
            tsp_insts: TSP instances (n_tsps, max_tsp_len, 2)
            n_tsps_per_route: Number of TSP instances per route
        """
        tsp_pis = []
        n_tsps_per_route = []

        for route in routes:
            start = 0
            sub_route_count = 0
            for idx, node in enumerate(route):
                if idx == 0:
                    continue
                if node == 0:  # Depot marks end of sub-route
                    if route[idx - 1] != 0:
                        tsp_pis.append(route[start:idx])
                        sub_route_count += 1
                    start = idx
            n_tsps_per_route.append(sub_route_count)

        if len(tsp_pis) == 0:
            return None, n_tsps_per_route

        # Pad to same length
        max_tsp_len = max(len(pi) for pi in tsp_pis)
        max_tsp_len = max(min_reviser_size, max_tsp_len)

        padded_tsp_pis = []
        for pi in tsp_pis:
            padded_pi = torch.nn.functional.pad(pi, (0, max_tsp_len - len(pi)), mode='constant', value=0)
            padded_tsp_pis.append(padded_pi)

        padded_tsp_pis = torch.stack(padded_tsp_pis)
        tsp_insts = coors[padded_tsp_pis]

        return tsp_insts, n_tsps_per_route

    def _sum_cost(self, costs: torch.Tensor, n_tsps_per_route: List[int]) -> torch.Tensor:
        """Sum costs per route."""
        if not isinstance(costs, torch.Tensor):
            costs = torch.tensor(costs)

        ret = []
        start = 0
        for n in n_tsps_per_route:
            if n > 0:
                ret.append(costs[start:start + n].sum())
            else:
                ret.append(torch.tensor(0.0, device=costs.device))
            start += n
        return torch.stack(ret)

    @torch.no_grad()
    def eval_routes(
        self,
        coords: torch.Tensor,
        routes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Evaluate routes using Reviser (following GLOP's eval function).

        Args:
            coords: Coordinates (n_nodes+1, 2) including depot at index 0
            routes: Route tensor (width, max_route_len) with node indices

        Returns:
            costs: Route costs (width,)
        """
        original_device = coords.device

        # Convert routes to TSP instances
        tsp_insts, n_tsps_per_route = self._trans_tsp(coords, routes, min(self.revision_lens))

        if tsp_insts is None or sum(n_tsps_per_route) == 0:
            # No valid routes
            return torch.zeros(routes.size(0), device=original_device)

        tsp_insts = tsp_insts.to(self.device)
        self.opts.eval_batch_size = tsp_insts.size(0)
        p_size = tsp_insts.size(1)

        # Initialize with random insertion (following GLOP exactly)
        seeds = tsp_insts
        order = torch.arange(p_size)

        if HAS_RANDOM_INSERTION:
            pi_all = random_insertion_parallel(seeds.cpu(), order)
            pi_all = torch.tensor(pi_all.astype(np.int64), device=self.device).reshape(-1, p_size)
            seeds = seeds.gather(1, pi_all.unsqueeze(-1).expand_as(seeds))

        # Apply Reviser improvements
        _, costs_revised = reconnect(
            get_cost_func=self.get_cost_func,
            batch=seeds,
            opts=self.opts,
            revisers=self.revisers,
        )

        # Sum costs per route
        route_costs = self._sum_cost(costs_revised, n_tsps_per_route)

        return route_costs.to(original_device)

    def eval_routes_batch(
        self,
        coords_list: List[torch.Tensor],
        routes_list: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """
        Evaluate routes for multiple instances.

        Args:
            coords_list: List of coordinate tensors
            routes_list: List of route tensors

        Returns:
            costs_list: List of cost tensors
        """
        return [
            self.eval_routes(coords, routes)
            for coords, routes in zip(coords_list, routes_list)
        ]
