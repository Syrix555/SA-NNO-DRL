import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from nets.attention_model import AttentionModel
from problems.op.problem_op import OP
from problems.op.state_op import StateOP
from train import augment


@pytest.fixture
def sa():
    path = Path(__file__).resolve().parents[1] / "SA-NNO-DRL.py"
    spec = importlib.util.spec_from_file_location("sa_padding_tests", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.device = torch.device("cpu")
    return module


def padded_batch():
    return {
        "loc": torch.tensor([[[.6, .5], [.5, .7], [.2, .5], [.5, .5]]]),
        "prize": torch.tensor([[1., 1., 1., 0.]]),
        "depot": torch.tensor([[.5, .5]]),
        "max_length": torch.tensor([2.]),
        "valid_nodes": torch.tensor([[True, True, True, False]]),
    }


@pytest.mark.parametrize("visited_dtype", [torch.uint8, torch.int64])
def test_padding_is_excluded_from_starts_and_actions(visited_dtype):
    batch = padded_batch()
    state = StateOP.initialize(batch, pomo_size=2, visited_dtype=visited_dtype)

    assert state.get_k_nearest_neighbor().tolist() == [[1, 2, 3, 0]]
    assert state.get_mask()[0, :, 4].all()
    assert not state.get_mask()[0, :, 0].any()

    state = state.update(torch.tensor([[1, 2]]))
    assert state.get_mask()[0, :, 4].all()
    assert not state.get_mask()[0, :, 0].any()


def test_zero_prize_real_target_at_hub_is_not_padding():
    batch = padded_batch()
    batch["valid_nodes"][:] = True
    state = StateOP.initialize(batch, pomo_size=1)

    assert state.get_k_nearest_neighbor()[0, 0].item() == 4
    assert not state.get_mask()[0, 0, 4]

    del batch["valid_nodes"]
    legacy = StateOP.initialize(batch, pomo_size=1)
    assert torch.equal(legacy.get_mask(), state.get_mask())
    assert torch.equal(legacy.get_k_nearest_neighbor(), state.get_k_nearest_neighbor())


def test_knn_places_feasible_real_nodes_before_unreachable_and_padding():
    batch = padded_batch()
    batch["max_length"][:] = .5
    state = StateOP.initialize(batch, pomo_size=1)

    assert state.get_k_nearest_neighbor().tolist() == [[1, 2, 0, 0]]


def test_padding_mask_follows_state_row_reordering_and_duplication():
    batch = {key: value.repeat(2, *([1] * (value.ndim - 1)))
             for key, value in padded_batch().items()}
    batch["valid_nodes"][1] = torch.tensor([False, False, False, True])
    state = StateOP.initialize(batch, pomo_size=2)
    rows = torch.tensor([1, 0, 1])
    reordered = state[rows]

    assert reordered.get_k_nearest_neighbor().tolist() == [
        [4, 0, 0, 0], [1, 2, 3, 0], [4, 0, 0, 0],
    ]
    assert torch.equal(reordered.get_mask(), state.get_mask()[rows])


def test_augmentation_repeats_valid_nodes_with_each_geometry():
    batch = padded_batch()
    valid = batch["valid_nodes"].clone()
    result = augment(batch, 4, "op")

    assert result["valid_nodes"].dtype == torch.bool
    assert torch.equal(result["valid_nodes"], valid.repeat(4, 1))
    assert result["valid_nodes"].shape == result["prize"].shape


@pytest.fixture
def model():
    torch.manual_seed(1234)
    model = AttentionModel(16, 16, OP, n_encode_layers=1, n_heads=4).eval()
    model.set_decode_type("greedy")
    return model


def test_decoder_never_visits_padding(model):
    batch = padded_batch()
    with torch.no_grad():
        cost, _, routes = model(batch, return_pi=True)

    assert routes[0, 0, 0].item() == 1
    assert not (routes == 4).any()
    assert torch.isfinite(cost).all()


def test_decoder_handles_fewer_valid_starts_and_empty_uav(model):
    # Four parallel starts, but only one feasible real target in row 0 and
    # no real targets in row 1.
    batch = {
        "loc": torch.full((2, 15, 2), .5),
        "prize": torch.zeros(2, 15),
        "depot": torch.full((2, 2), .5),
        "max_length": torch.full((2,), 2.),
        "valid_nodes": torch.zeros(2, 15, dtype=torch.bool),
    }
    batch["loc"][0, 0] = torch.tensor([.6, .5])
    batch["prize"][0, 0] = 1.
    batch["valid_nodes"][0, 0] = True
    with torch.no_grad():
        cost, probabilities, routes = model(batch, return_pi=True)

    assert routes[:, :, 0].tolist() == [[1, 0, 0, 0], [0, 0, 0, 0]]
    assert not (routes > 1).any()
    assert cost.tolist() == [[-1., 0., 0., 0.], [0., 0., 0., 0.]]
    assert torch.isfinite(probabilities).all()


@pytest.mark.parametrize("node_count", [0, 1, 2, 4])
def test_all_empty_or_unreachable_routes_keep_candidate_cost_shape(model, node_count):
    batch = {
        "loc": torch.full((2, node_count, 2), .9),
        "prize": torch.ones(2, node_count),
        "depot": torch.full((2, 2), .5),
        "max_length": torch.full((2,), .1),
    }
    with torch.no_grad():
        cost, _, routes = model(batch, return_pi=True)

    assert routes.shape == (2, max(1, (node_count + 1) // 4), 1)
    assert cost.shape == routes.shape[:2]
    assert (routes == 0).all()
    assert (cost == 0).all()


def test_validation_checks_expected_reward_with_float_tolerance(sa):
    args = (
        [[.6, .5], [.5, .6]], [.5, .5], [.1, .2], 2.,
        [np.array([0, 1])], [np.array([0, 1])],
    )
    reward, length = sa.validate_allocation(*args, expected_prize=.3)
    assert reward == pytest.approx(.3)
    assert length < 2.
    with pytest.raises(ValueError, match="reward"):
        sa.validate_allocation(*args, expected_prize=1.3)
    with pytest.raises(ValueError, match="reward"):
        sa.validate_allocation(*args, expected_prize=float("nan"))


def test_sa_rejects_inconsistent_decoder_reward(sa, monkeypatch):
    def wrong_reward(dataset, uav_num):
        return np.ones(uav_num), np.zeros((uav_num, 1), dtype=np.int64)

    monkeypatch.setattr(sa, "cal_fitness", wrong_reward)
    with pytest.raises(ValueError, match="reward"):
        sa.sa_allocate([[.6, .5], [.5, .6]], [.5, .5], [1., 1.], 2., 2)


def test_sa_tags_padding_after_redistribution(sa, monkeypatch):
    padding_counts = []

    def controlled_decoder(dataset, uav_num):
        counts, fitness = [], []
        routes = np.zeros((uav_num, 1), dtype=np.int64)
        for i, record in enumerate(dataset):
            valid = record["valid_nodes"]
            assert valid.dtype == torch.bool
            counts.append(int((~valid).sum()))
            assert (record["loc"][~valid] == record["depot"]).all()
            assert (record["prize"][~valid] == 0).all()
            real = torch.where(valid)[0]
            if i == 0 and len(real):
                routes[i, 0] = int(real[0]) + 1
                fitness.append(float(record["prize"][real[0]]))
            else:
                fitness.append(0.)
        padding_counts.append(counts)
        return np.asarray(fitness, dtype=np.float32), routes

    monkeypatch.setattr(sa, "cal_fitness", controlled_decoder)
    angles = np.linspace(-np.pi + .1, np.pi - .1, 8)
    task = np.stack([.5 + .1 * np.cos(angles), .5 + .1 * np.sin(angles)], -1)
    fitness, _, _ = sa.sa_allocate(task, [.5, .5], np.ones(8), 2., 2)

    assert padding_counts[0] == [0, 0]
    assert any(sum(counts) > 0 for counts in padding_counts[1:])
    assert fitness == 1.


def test_sa_with_real_decoder_and_augmentation(sa, model, monkeypatch):
    monkeypatch.setattr(sa, "model", model, raising=False)
    task = np.array([[.6, .5], [.5, .6], [.4, .5], [.5, .4], [.6, .6]])
    prize = np.ones(len(task))
    fitness, assignments, routes = sa.sa_allocate(task, [.5, .5], prize, 2., 2)
    reward, _ = sa.validate_allocation(
        task, [.5, .5], prize, 2., assignments, routes, expected_prize=fitness,
    )

    assert reward == pytest.approx(float(fitness))
    assert reward > 0


def test_sa_with_no_reachable_targets_and_augmentation(sa, model, monkeypatch):
    monkeypatch.setattr(sa, "model", model, raising=False)
    task = np.array([[.9, .9], [.1, .1], [.1, .9], [.9, .1], [.8, .8]])
    prize = np.ones(len(task))
    fitness, assignments, routes = sa.sa_allocate(task, [.5, .5], prize, .1, 2)

    assert fitness == 0.
    assert all(len(route) == 0 for route in routes)
    reward, length = sa.validate_allocation(
        task, [.5, .5], prize, .1, assignments, routes, expected_prize=fitness,
    )
    assert reward == length == 0.
