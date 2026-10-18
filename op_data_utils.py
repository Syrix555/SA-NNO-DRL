"""Shared OP data contract, independent of PyTorch and model imports."""

from collections.abc import Mapping
from pathlib import Path
import pickle

import numpy as np


def _validate_fields(loc, prize, depot, max_length):
    loc = np.asarray(loc, dtype=np.float32)
    prize = np.asarray(prize, dtype=np.float32)
    depot = np.asarray(depot, dtype=np.float32)
    max_length = np.asarray(max_length, dtype=np.float32)
    if loc.ndim != 2 or loc.shape[1] != 2 or loc.shape[0] == 0:
        raise ValueError("loc must have shape [N, 2], with N > 0")
    if prize.shape != (len(loc),):
        raise ValueError("prize must have shape [N], matching loc")
    if depot.shape == (1, 2):
        depot = depot[0]
    if depot.shape != (2,):
        raise ValueError("depot must have shape [2] or [1, 2]")
    if max_length.shape != () or max_length <= 0:
        raise ValueError("max_length must be a positive scalar")
    fields = dict(loc=loc, prize=prize, depot=depot, max_length=max_length)
    if not all(np.isfinite(value).all() for value in fields.values()):
        raise ValueError("OP fields must contain only finite values")
    if (prize < 0).any():
        raise ValueError("prize must be nonnegative")
    return fields


def normalize_op_instance(instance):
    """Accept dictionaries or either historical four-field tuple ordering.

    Tuple formats: (loc, prize, depot, max_length) and
    (depot, loc, prize, max_length). Ambiguous records are rejected.
    """
    if isinstance(instance, Mapping):
        try:
            return _validate_fields(*(instance[key] for key in
                                      ("loc", "prize", "depot", "max_length")))
        except KeyError as exc:
            raise ValueError(f"Missing OP field: {exc.args[0]}") from exc
    if not isinstance(instance, (tuple, list)) or len(instance) != 4:
        raise ValueError("OP records must be dictionaries or four-field tuples")
    first, second, third, length = instance
    candidates = []
    for fields in ((first, second, third, length), (second, third, first, length)):
        try:
            candidates.append(_validate_fields(*fields))
        except (ValueError, TypeError):
            continue
    if len(candidates) != 1:
        raise ValueError("Invalid or ambiguous OP tuple; use named fields")
    return candidates[0]


def load_op_dataset(filename, num_samples=None, offset=0):
    """Load trusted local pickle data, retaining each record's length budget."""
    if offset < 0 or (num_samples is not None and num_samples < 0):
        raise ValueError("offset and num_samples must be nonnegative")
    path = Path(filename)
    if path.suffix != ".pkl":
        raise ValueError("OP dataset filename must end in .pkl")
    with path.open("rb") as handle:
        records = pickle.load(handle)
    end = None if num_samples is None else offset + num_samples
    data = []
    for index, record in enumerate(records[offset:end], start=offset):
        try:
            data.append(normalize_op_instance(record))
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{path}: invalid OP record {index}: {exc}") from exc
    return data
