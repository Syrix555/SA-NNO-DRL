import warnings
import pickle

import torch
import numpy as np
import os
import json
from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Pool
import torch.nn.functional as F


def load_problem(name):
    from problems import TSP, CVRP, SDVRP, OP, PCTSPDet, PCTSPStoch
    problem = {
        'tsp': TSP,
        'cvrp': CVRP,
        'sdvrp': SDVRP,
        'op': OP,
        'pctsp_det': PCTSPDet,
        'pctsp_stoch': PCTSPStoch,
    }.get(name, None)
    assert problem is not None, "Currently unsupported problem: {}!".format(name)
    return problem


def torch_load_cpu(load_path, allow_legacy=False):
    """Load a checkpoint on CPU using PyTorch's restricted unpickler by default."""
    try:
        return torch.load(load_path, map_location='cpu', weights_only=not allow_legacy)
    except (pickle.UnpicklingError, RuntimeError) as exc:
        if allow_legacy:
            raise
        raise RuntimeError(
            f"Cannot safely load checkpoint {load_path}. "
            "If this is a trusted legacy checkpoint, rerun with --allow_legacy_checkpoint."
        ) from exc


def move_to(var, device):
    if isinstance(var, dict):
        return {k: move_to(v, device) for k, v in var.items()}
    return var.to(device)


def load_args(filename):
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


def load_model(path, epoch=None, allow_legacy=False):
    from nets.attention_model import AttentionModel
    from nets.pointer_network import PointerNetwork

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

    problem = load_problem(args['problem'])

    model_class = {
        'attention': AttentionModel,
        'pointer': PointerNetwork
    }.get(args.get('model', 'attention'), None)
    assert model_class is not None, "Unknown model: {}".format(model_class)

    model = model_class(
        args['embedding_dim'],
        args['hidden_dim'],
        problem,
        n_encode_layers=args['n_encode_layers'],
        mask_inner=True,
        mask_logits=True,
        normalization=args['normalization'],
        tanh_clipping=args['tanh_clipping'],
        checkpoint_encoder=args.get('checkpoint_encoder', False),
        shrink_size=args.get('shrink_size', None)
    )
    print('  [*] Loading model from {}'.format(model_filename))
    load_data = torch_load_cpu(model_filename, allow_legacy=allow_legacy)
    if not isinstance(load_data, dict) or 'model' not in load_data:
        raise ValueError(f"Checkpoint {model_filename} does not contain a model state_dict")
    model.load_state_dict(load_data['model'], strict=True)

    model.eval()  # Put in eval mode

    return model, args


def parse_softmax_temperature(raw_temp):
    # Load from file
    if os.path.isfile(raw_temp):
        return np.loadtxt(raw_temp)[-1, 0]
    return float(raw_temp)


def run_all_in_pool(func, directory, dataset, opts, use_multiprocessing=True):
    # # Test
    # res = func((directory, 'test', *dataset[0]))
    # return [res]

    num_cpus = os.cpu_count() if opts.cpus is None else opts.cpus

    w = len(str(len(dataset) - 1))
    offset = getattr(opts, 'offset', None)
    if offset is None:
        offset = 0
    ds = dataset[offset:(offset + opts.n if opts.n is not None else len(dataset))]
    pool_cls = (Pool if use_multiprocessing and num_cpus > 1 else ThreadPool)
    with pool_cls(num_cpus) as pool:
        results = list(tqdm(pool.imap(
            func,
            [
                (
                    directory,
                    str(i + offset).zfill(w),
                    *problem
                )
                for i, problem in enumerate(ds)
            ]
        ), total=len(ds), mininterval=opts.progress_bar_mininterval))

    failed = [str(i + offset) for i, res in enumerate(results) if res is None]
    assert len(failed) == 0, "Some instances failed: {}".format(" ".join(failed))
    return results, num_cpus


def do_batch_rep(v, n):
    if isinstance(v, dict):
        return {k: do_batch_rep(v_, n) for k, v_ in v.items()}
    elif isinstance(v, list):
        return [do_batch_rep(v_, n) for v_ in v]
    elif isinstance(v, tuple):
        return tuple(do_batch_rep(v_, n) for v_ in v)

    return v[None, ...].expand(n, *v.size()).contiguous().view(-1, *v.size()[1:])


def reduce_pomo_candidates(costs, sequences=None):
    """Select the lowest-cost POMO candidate independently per instance.

    ``costs`` must be ``[batch, pomo]``. If sequences are supplied they must
    be ``[batch, pomo, steps]`` and are gathered with the same candidate index.
    Orienteering costs are negative collected prizes, so ``min`` is the desired
    objective in both training and evaluation.
    """
    if costs.ndim != 2:
        raise ValueError(f"costs must have shape [batch, pomo], got {tuple(costs.shape)}")
    best_costs, best_idx = costs.min(dim=1)
    if sequences is None:
        return best_costs, None
    if sequences.ndim != 3 or sequences.shape[:2] != costs.shape:
        raise ValueError(
            f"sequences must have shape [batch, pomo, steps] matching costs, got {tuple(sequences.shape)}"
        )
    best_sequences = sequences.gather(1, best_idx[:, None, None].expand(-1, 1, sequences.size(-1))).squeeze(1)
    return best_costs, best_sequences


def sample_many(inner_func, get_cost_func, input, batch_rep=1, iter_rep=1):
    """
    :param input: (batch_size, graph_size, node_dim) input node features
    :return:
    """
    if batch_rep < 1 or iter_rep < 1:
        raise ValueError("batch_rep and iter_rep must be positive")

    def batch_size(value):
        if isinstance(value, dict):
            return batch_size(next(iter(value.values())))
        if isinstance(value, (tuple, list)):
            return batch_size(value[0])
        return value.size(0)

    source_batch_size = batch_size(input)
    input = do_batch_rep(input, batch_rep)
    costs = []
    pis = []
    for i in range(iter_rep):
        _log_p, pi = inner_func(input)
        cost, mask = get_cost_func(input, pi)

        if cost.ndim != 2 or pi.ndim != 3:
            raise ValueError("sample_many expects model costs [batch, pomo] and routes [batch, pomo, steps]")
        best_cost, best_pi = reduce_pomo_candidates(cost, pi)
        costs.append(best_cost.view(batch_rep, source_batch_size).transpose(0, 1))
        pis.append(best_pi.view(batch_rep, source_batch_size, pi.size(-1)).transpose(0, 1))

    max_length = max(pi.size(-1) for pi in pis)
    # (batch_size, iter_rep, max_length) => (batch_size, batch_rep * iter_rep, max_length)
    pis = torch.cat(
        [F.pad(pi, (0, max_length - pi.size(-1))) for pi in pis],
        1
    )  # .view(embeddings.size(0), batch_rep * iter_rep, max_length)
    costs = torch.cat(costs, 1)

    # Select the best sampled candidate per original instance.
    mincosts, argmincosts = costs.min(-1)
    # (batch_size, minlength)
    minpis = pis[torch.arange(pis.size(0), out=argmincosts.new()), argmincosts]

    return minpis, mincosts
