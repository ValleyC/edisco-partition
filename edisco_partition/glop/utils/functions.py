"""
Core functions from GLOP for loading and using Revisers.

This module provides:
- load_model: Load pretrained Reviser models
- load_problem: Load problem definitions (TSP, LOCAL)
- reconnect: Main function for improving TSP tours using Revisers
"""
import torch
import numpy as np
import os
import json
import sys
import time


def _setup_module_aliases():
    """
    Setup module aliases so torch.load can find GLOP classes.

    The pretrained models reference 'nets.attention_local.AttentionModel' etc.,
    but our package structure is 'edisco_partition.glop.nets.attention_local'.
    This creates aliases to bridge the gap.
    """
    # Import our modules
    from .. import nets
    from ..nets import attention_local, graph_encoder
    from .. import problems
    from ..problems import local, tsp
    from ..problems.local import problem_local, state_local
    from ..problems.tsp import problem_tsp
    from .. import utils
    from . import boolmask, tensor_functions

    # Create module aliases for pickle/torch.load compatibility
    sys.modules['nets'] = nets
    sys.modules['nets.attention_local'] = attention_local
    sys.modules['nets.graph_encoder'] = graph_encoder
    sys.modules['problems'] = problems
    sys.modules['problems.local'] = local
    sys.modules['problems.local.problem_local'] = problem_local
    sys.modules['problems.local.state_local'] = state_local
    sys.modules['problems.tsp'] = tsp
    sys.modules['problems.tsp.problem_tsp'] = problem_tsp
    sys.modules['utils'] = utils
    sys.modules['utils.boolmask'] = boolmask
    sys.modules['utils.tensor_functions'] = tensor_functions


# Setup aliases on import
_setup_module_aliases()


def load_problem(name):
    """Load problem class by name."""
    from ..problems import TSP, LOCAL
    problem = {
        'local': LOCAL,
        'tsp': TSP,
    }.get(name, None)
    assert problem is not None, "Currently unsupported problem: {}!".format(name)
    return problem


def torch_load_cpu(load_path):
    """Load checkpoint to CPU with weights_only=False for custom classes."""
    return torch.load(load_path, map_location=lambda storage, loc: storage, weights_only=False)


def move_to(var, device):
    if isinstance(var, dict):
        return {k: move_to(v, device) for k, v in var.items()}
    return var.to(device)


def load_args(filename):
    """Load model args from JSON file."""
    with open(filename, 'r') as f:
        args = json.load(f)

    # Backwards compatibility
    if 'data_distribution' not in args:
        args['data_distribution'] = None
        probl, *dist = args['problem'].split("_")
        if probl == "op":
            args['problem'] = probl
            args['data_distribution'] = dist[0]
    return args


def load_model(path, epoch=None, is_local=True):
    """
    Load a pretrained Reviser model.

    Args:
        path: Path to model directory or checkpoint file
        epoch: Epoch number to load (if path is directory)
        is_local: Whether to load LOCAL problem model (for SHPP)

    Returns:
        model: Loaded model in eval mode
        args: Model arguments
    """
    from ..nets.attention_local import AttentionModel

    if os.path.isfile(path):
        model_filename = path
        path = os.path.dirname(model_filename)
    elif os.path.isdir(path):
        if epoch is None:
            epoch = max(
                int(os.path.splitext(filename)[0].split("-")[1])
                for filename in os.listdir(path)
                if os.path.splitext(filename)[1] == '.pt'
            )
        model_filename = os.path.join(path, 'epoch-{}.pt'.format(epoch))
    else:
        assert False, "{} is not a valid directory or file".format(path)

    args = load_args(os.path.join(path, 'args.json'))

    if is_local:
        model = AttentionModel(
            args['embedding_dim'],
            args['hidden_dim'],
            load_problem('local'),
            n_encode_layers=args['n_encode_layers'],
            mask_inner=True,
            mask_logits=True,
            normalization=args['normalization'],
            tanh_clipping=args['tanh_clipping'],
            checkpoint_encoder=args.get('checkpoint_encoder', False),
            shrink_size=args.get('shrink_size', None),
        )
    else:
        raise NotImplementedError("Only LOCAL problem supported")

    # Load model weights
    load_data = torch_load_cpu(model_filename)
    model.load_state_dict({**model.state_dict(), **load_data.get('model', {})})

    model.eval()  # Put in eval mode

    return model, args


def coordinate_transformation(x):
    """
    Transform coordinates for Reviser input.
    Normalizes to [0,1] range with x-axis being the longer dimension.
    """
    input = x.clone()
    max_x, _ = input[:, :, 0].max(dim=1)
    max_y, _ = input[:, :, 1].max(dim=1)
    min_x, _ = input[:, :, 0].min(dim=1)
    min_y, _ = input[:, :, 1].min(dim=1)

    diff_x = max_x - min_x
    diff_y = max_y - min_y
    xy_exchanged = diff_y > diff_x

    # Shift to zero
    input[:, :, 0] -= min_x.unsqueeze(-1)
    input[:, :, 1] -= min_y.unsqueeze(-1)

    # Exchange coordinates for those diff_y > diff_x
    input[xy_exchanged, :, 0], input[xy_exchanged, :, 1] = \
        input[xy_exchanged, :, 1].clone(), input[xy_exchanged, :, 0].clone()

    # Scale to (0, 1)
    scale_degree = torch.max(diff_x, diff_y)
    scale_degree = scale_degree.view(input.shape[0], 1, 1)
    input = input / (scale_degree + 1e-10)
    return input


def decomposition(seeds, coordinate_dim, revision_len, offset, shift_len=1):
    """Decompose a tour into subtours for revision."""
    # Change decomposition point
    seeds = torch.cat([seeds[:, shift_len:], seeds[:, :shift_len]], 1)

    if offset != 0:
        decomposed_seeds = seeds[:, :-offset]
        offset_seeds = seeds[:, -offset:]
    else:
        decomposed_seeds = seeds
        offset_seeds = None

    decomposed_seeds = decomposed_seeds.reshape(-1, revision_len, coordinate_dim)
    return decomposed_seeds, offset_seeds


def revision(opts, revision_cost_func, reviser, decomposed_seeds, original_subtour, iter=None, embeddings=None):
    """Revise subtours using the Reviser model."""
    reviser_size = original_subtour.shape[0]
    init_cost = revision_cost_func(decomposed_seeds, original_subtour)

    # Coordinate transformation
    transformed_seeds = coordinate_transformation(decomposed_seeds)

    # Augmentation
    if not opts.no_aug:
        seed2 = torch.cat((1 - transformed_seeds[:, :, [0]], transformed_seeds[:, :, [1]]), dim=2)
        seed3 = torch.cat((transformed_seeds[:, :, [0]], 1 - transformed_seeds[:, :, [1]]), dim=2)
        seed4 = torch.cat((1 - transformed_seeds[:, :, [0]], 1 - transformed_seeds[:, :, [1]]), dim=2)
        augmented_seeds = torch.cat((transformed_seeds, seed2, seed3, seed4), dim=0)
    else:
        augmented_seeds = transformed_seeds

    if iter is None:
        cost_revised1, sub_tour1, cost_revised2, sub_tour2 = reviser(augmented_seeds, return_pi=True)
    elif iter == 0:
        cost_revised1, sub_tour1, cost_revised2, sub_tour2, embeddings = reviser(augmented_seeds, return_pi=True, return_embedding=True)
    else:
        cost_revised1, sub_tour1, cost_revised2, sub_tour2 = reviser(augmented_seeds, return_pi=True, embeddings=embeddings)

    if not opts.no_aug:
        _, better_tour_idx = torch.cat([cost_revised1, cost_revised2], dim=0).reshape(8, -1).min(dim=0)
        sub_tour = torch.cat([sub_tour1, sub_tour2], dim=0).reshape(8, -1, reviser_size)[better_tour_idx, torch.arange(sub_tour1.shape[0] // 4), :]
    else:
        _, better_tour_idx = torch.stack((cost_revised1, cost_revised2)).min(dim=0)
        sub_tour = torch.stack((sub_tour1, sub_tour2))[better_tour_idx, torch.arange(sub_tour1.shape[0])]

    cost_revised, _ = reviser.problem.get_costs(decomposed_seeds, sub_tour)
    reduced_cost = init_cost - cost_revised

    sub_tour[reduced_cost < 0] = original_subtour
    decomposed_seeds = decomposed_seeds.gather(1, sub_tour.unsqueeze(-1).expand_as(decomposed_seeds))

    if embeddings is not None:
        if not opts.no_aug:
            embeddings = embeddings.gather(1, sub_tour.repeat(4, 1).unsqueeze(-1).expand_as(embeddings))
        else:
            embeddings = embeddings.gather(1, sub_tour.unsqueeze(-1).expand_as(embeddings))

    return decomposed_seeds, embeddings


def LCP_TSP(seeds, cost_func, reviser, revision_len, revision_iter, opts, shift_len):
    """Local Construction Policy for TSP improvement."""
    batch_size, num_nodes, coordinate_dim = seeds.shape
    offset = num_nodes % revision_len
    embeddings = None  # Used only in case problem_size == revision_len for efficiency

    for i in range(revision_iter):
        decomposed_seeds, offset_seed = decomposition(
            seeds, coordinate_dim, revision_len, offset, shift_len
        )

        original_subtour = torch.arange(0, revision_len, dtype=torch.long).to(decomposed_seeds.device)

        if revision_len == num_nodes:
            decomposed_seeds_revised, embeddings = revision(
                opts, cost_func, reviser, decomposed_seeds, original_subtour, iter=i, embeddings=embeddings
            )
            embeddings = torch.cat([embeddings[:, shift_len:], embeddings[:, :shift_len]], 1)  # Roll the embeddings
        else:
            decomposed_seeds_revised, _ = revision(
                opts, cost_func, reviser, decomposed_seeds, original_subtour
            )

        seeds = decomposed_seeds_revised.reshape(batch_size, -1, coordinate_dim)
        if offset_seed is not None:
            seeds = torch.cat([seeds, offset_seed], dim=1)

    return seeds


def reconnect(get_cost_func, batch, opts, revisers):
    """
    Reconnect/improve TSP solutions using Revisers.

    Args:
        get_cost_func: Function to compute TSP costs
        batch: TSP instances (batch_size, num_nodes, 2)
        opts: Options with revision_lens, revision_iters, no_aug, no_prune, eval_batch_size
        revisers: List of Reviser models

    Returns:
        seed: Improved TSP solutions
        cost_revised: Final costs
    """
    seed = batch
    problem_size = seed.size(1)

    if len(revisers) == 0:
        cost_revised = (seed[:, 1:] - seed[:, :-1]).norm(p=2, dim=2).sum(1) + (seed[:, 0] - seed[:, -1]).norm(p=2, dim=1)
        return seed, cost_revised

    for revision_id in range(len(revisers)):
        assert opts.revision_lens[revision_id] <= seed.size(1)
        shift_len = max(opts.revision_lens[revision_id] // opts.revision_iters[revision_id], 1)
        seed = LCP_TSP(
            seed,
            get_cost_func,
            revisers[revision_id],
            opts.revision_lens[revision_id],
            opts.revision_iters[revision_id],
            opts=opts,
            shift_len=shift_len
        )
        cost_revised = (seed[:, 1:] - seed[:, :-1]).norm(p=2, dim=2).sum(1) + (seed[:, 0] - seed[:, -1]).norm(p=2, dim=1)

        if revision_id == 0 and not opts.no_prune:  # Eliminate underperforming ones after first round
            cost_revised, cost_revised_minidx = cost_revised.reshape(-1, opts.eval_batch_size).min(0)
            seed = seed.reshape(-1, opts.eval_batch_size, seed.shape[-2], 2)[cost_revised_minidx, torch.arange(opts.eval_batch_size)]

    if opts.no_prune:
        cost_revised, cost_revised_minidx = cost_revised.reshape(-1, opts.eval_batch_size).min(0)
        seed = seed.reshape(-1, opts.eval_batch_size, seed.shape[-2], 2)[cost_revised_minidx, torch.arange(opts.eval_batch_size)]

    return seed, cost_revised
