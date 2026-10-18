"""Startup validation without loading checkpoints or generating datasets."""

from pathlib import Path
import re


def resolve_model_path(path):
    """Resolve a checkpoint and sibling args.json without deserializing it."""
    path = Path(path)
    if path.is_dir():
        candidates = []
        for candidate in path.glob('epoch-*.pt'):
            match = re.fullmatch(r'epoch-(\d+)\.pt', candidate.name)
            if match and candidate.is_file():
                candidates.append((int(match[1]), candidate))
        if not candidates:
            raise FileNotFoundError(f"No epoch-*.pt checkpoint in {path}; train a model or supply --model FILE")
        path = max(candidates, key=lambda item: item[0])[1]
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}; train a model or supply an existing checkpoint")
    args_path = path.parent / 'args.json'
    if not args_path.is_file():
        raise FileNotFoundError(f"Model configuration not found: {args_path}; keep args.json beside the checkpoint")
    return path


def resolve_device(requested='auto', no_cuda=False):
    import torch

    if no_cuda:
        if requested not in ('auto', 'cpu'):
            raise ValueError('--no_cuda conflicts with an explicit CUDA device')
        requested = 'cpu'
    if requested == 'auto':
        requested = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(requested)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('Only cpu and cuda devices are supported')
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError('CUDA is unavailable; use --device cpu or install the cu130 extra')
        if device.index is None:
            device = torch.device('cuda:0')
        if device.index >= torch.cuda.device_count():
            raise ValueError(f'CUDA device index out of range: {device}')
    return device
