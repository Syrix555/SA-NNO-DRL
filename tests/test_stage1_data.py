import numpy as np
import pytest
import torch

from op_data_utils import load_op_dataset, normalize_op_instance
from utils.functions import reduce_pomo_candidates, sample_many


def test_load_repository_op_data_uses_common_contract():
    records = load_op_dataset("op_data/unif/op_unif300_len2_seed12345.pkl", num_samples=2)

    assert len(records) == 2
    assert records[0]["loc"].shape == (300, 2)
    assert records[0]["prize"].shape == (300,)
    assert records[0]["depot"].shape == (2,)
    assert records[0]["loc"].dtype == np.float32


@pytest.mark.parametrize(
    "record",
    [
        (np.zeros((4, 2)), np.ones(4), np.array([0.5, 0.5]), 2.0),
        ([0.5, 0.5], np.zeros((4, 2)).tolist(), np.ones(4).tolist(), 2.0),
    ],
)
def test_normalize_historical_tuple_orders(record):
    normalized = normalize_op_instance(record)

    assert normalized["loc"].shape == (4, 2)
    assert normalized["prize"].shape == (4,)
    assert normalized["depot"].shape == (2,)


def test_normalize_rejects_invalid_prize_shape():
    with pytest.raises(ValueError, match="Invalid|prize"):
        normalize_op_instance((np.zeros((4, 2)), np.ones(3), [0.5, 0.5], 2.0))


def test_reduce_pomo_candidates_keeps_route_aligned_with_minimum_cost():
    costs = torch.tensor([[3.0, 1.0, 2.0], [0.5, 4.0, 0.25]])
    routes = torch.tensor([
        [[10, 11], [20, 21], [30, 31]],
        [[40, 41], [50, 51], [60, 61]],
    ])

    best_cost, best_route = reduce_pomo_candidates(costs, routes)

    assert torch.equal(best_cost, torch.tensor([1.0, 0.25]))
    assert torch.equal(best_route, torch.tensor([[20, 21], [60, 61]]))


def test_sample_many_selects_best_candidate_per_original_instance():
    def inner(batch):
        # batch_rep duplicates each source row; each row has two POMO candidates.
        routes = torch.tensor([[[1], [2]], [[3], [4]], [[5], [6]], [[7], [8]]])
        probabilities = torch.ones_like(routes, dtype=torch.float32)
        return probabilities, routes

    def costs(batch, routes):
        return routes.squeeze(-1).float(), None

    routes, costs_out = sample_many(inner, costs, torch.zeros(2, 1), batch_rep=2)

    assert torch.equal(routes, torch.tensor([[1], [3]]))
    assert torch.equal(costs_out, torch.tensor([1.0, 3.0]))
