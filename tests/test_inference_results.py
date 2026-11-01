import csv
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np
import pytest
import torch

from inference_results import InferenceResults, instance_result
from nets.attention_model import AttentionModel
from problems.op.problem_op import OP


REPO = Path(__file__).resolve().parents[1]


def sample_result(instance_id=0, empty=False):
    batch = {
        "loc": np.array([[.6, .5], [.5, .7], [.4, .5], [.5, .3]], dtype=np.float32),
        "depot": np.array([.5, .5], dtype=np.float32),
        "prize": np.array([.1, .2, .3, .4], dtype=np.float32),
        "max_length": np.float32(2),
    }
    return instance_result(
        instance_id, batch, np.float32(0 if empty else .6),
        np.float64(3 if empty else 1),
        [np.array([2, 0]), np.array([3, 1]), np.array([], dtype=int)],
        [[], [], []] if empty else [[1, 0], [1], []],
    )


def test_export_converts_local_routes_and_keeps_idle_uavs():
    record = sample_result()
    assert [r["visited_target_ids"] for r in record["routes"]] == [[0, 2], [1], []]
    assert record["routes"][0]["assigned_target_ids"] == [2, 0]
    assert [r["reward"] for r in record["routes"]] == pytest.approx([.4, .2, 0])
    assert [r["route_length"] for r in record["routes"]] == pytest.approx([.4, .4, 0])
    assert record["total_route_length"] == pytest.approx(.8)
    assert record["visited_target_count"] == 3
    assert json.loads(json.dumps(record, allow_nan=False)) == record


def test_snapshots_statistics_provenance_and_unique_directories(tmp_path):
    checkpoint = tmp_path / "epoch-9.pt"
    checkpoint.write_bytes(b"test placeholder; not loaded")
    args = tmp_path / "args.json"
    args.write_text('{"data_distribution": "const"}\n')
    dataset = tmp_path / "data.pkl"
    dataset.write_bytes(b"test dataset placeholder; not loaded")
    options = {
        "dataset": str(dataset), "model": str(tmp_path), "seed": None,
        "weight_source": "self-trained", "output_dir": str(tmp_path / "results"),
        "uav_num": 3, "num_samples": 30,
    }
    np_state, torch_state = np.random.get_state(), torch.get_rng_state()
    output = InferenceResults(options, checkpoint, torch.device("cpu"))
    assert np.array_equal(np_state[1], np.random.get_state()[1])
    assert np_state[2:] == np.random.get_state()[2:]
    assert torch.equal(torch_state, torch.get_rng_state())
    output.add(sample_result())
    summary = json.loads((output.directory / "summary.json").read_text())
    assert summary["status"] == "running"
    assert summary["completed_samples"] == 1
    assert summary["metrics"]["reward_std"] is None

    output.add(sample_result(1, empty=True))
    output.save("completed")
    summary = json.loads((output.directory / "summary.json").read_text())
    assert summary["status"] == "completed"
    assert summary["requested_samples"] == 30
    assert summary["completed_samples"] == 2
    assert summary["metrics"] == pytest.approx({
        "reward_mean": .3, "reward_std": .6 / np.sqrt(2),
        "duration_seconds_mean": 2., "duration_seconds_std": np.sqrt(2),
    })
    assert summary["metadata"]["checkpoint"]["path"] == str(checkpoint)
    assert summary["metadata"]["checkpoint"]["sha256"] == hashlib.sha256(
        checkpoint.read_bytes(),
    ).hexdigest()
    assert (output.directory / "model_args.json").read_bytes() == args.read_bytes()
    with (output.directory / "instances.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    assert rows[0]["model"] == str(checkpoint)
    assert rows[0]["seed"] == ""
    assert float(rows[0]["reward"]) == pytest.approx(.6)
    routes = json.loads((output.directory / "routes.json").read_text())
    assert routes["seed"] is None
    assert routes["instances"][1]["routes"][0]["visited_target_ids"] == []

    second = InferenceResults(options, checkpoint, torch.device("cpu"))
    assert second.directory != output.directory
    assert json.loads((output.directory / "summary.json").read_text()) == summary
    broken = sample_result()
    broken["reward"] = 10.
    with pytest.raises(ValueError, match="reward"):
        second.add(broken)
    assert second.records == []


def test_cpu_cli_exports_real_inference_and_fig7(tmp_path):
    # Random weights exercise I/O and route validity, not trained model quality.
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    args = {
        "problem": "op", "embedding_dim": 16, "hidden_dim": 16,
        "n_encode_layers": 1, "normalization": "batch", "tanh_clipping": 10.,
        "data_distribution": "const",
    }
    (model_dir / "args.json").write_text(json.dumps(args))
    torch.manual_seed(1234)
    model = AttentionModel(16, 16, OP, n_encode_layers=1)
    checkpoint = model_dir / "epoch-9.pt"
    torch.save({"model": model.state_dict()}, checkpoint)
    dataset = tmp_path / "tiny.pkl"
    points = np.array([[.6, .5], [.5, .6], [.4, .5], [.5, .4], [.6, .6]])
    records = [
        (points, np.ones(5), [.5, .5], 2.),
        (points, np.ones(5), [.5, .5], .01),
    ]
    with dataset.open("wb") as stream:
        pickle.dump(records, stream)
    command = [
        sys.executable, str(REPO / "SA-NNO-DRL.py"),
        "--model", str(model_dir), "--dataset", str(dataset),
        "--num-samples", "30", "--uav-num", "2", "--device", "cpu",
        "--seed", "1234", "--output-dir", str(tmp_path / "results"),
        "--plot-dir", str(tmp_path / "plots"),
    ]
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               MPLCONFIGDIR=str(tmp_path / "mpl"))
    for _ in range(2):
        run = subprocess.run(command, cwd=REPO, env=env, capture_output=True,
                             text=True, timeout=60)
        assert run.returncode == 0, run.stdout + run.stderr
    directories = sorted((tmp_path / "results").iterdir())
    assert len(directories) == 2
    first_routes = None
    for directory in directories:
        summary = json.loads((directory / "summary.json").read_text())
        assert summary["status"] == "completed"
        assert summary["completed_samples"] == 2
        assert summary["requested_samples"] == 30
        assert summary["metadata"]["checkpoint"]["path"] == str(checkpoint)
        assert summary["options"]["seed"] == 1234
        routes = json.loads((directory / "routes.json").read_text())
        selected = [record["routes"] for record in routes["instances"]]
        if first_routes is None:
            first_routes = selected
        else:
            assert selected == first_routes
        rewards = []
        for record in routes["instances"]:
            visited = []
            reward = 0.
            for route in record["routes"]:
                ids = route["visited_target_ids"]
                assert set(ids) <= set(route["assigned_target_ids"])
                visited.extend(ids)
                coords = np.array([record["depot"], *np.array(record["loc"])[ids],
                                   record["depot"]])
                length = np.linalg.norm(np.diff(coords, axis=0), axis=1).sum()
                assert length == pytest.approx(route["route_length"], abs=1e-6)
                assert length <= record["max_length"] + 1e-5
                reward += np.array(record["prize"])[ids].sum()
            assert len(visited) == len(set(visited))
            assert reward == pytest.approx(record["reward"])
            rewards.append(reward)
        assert rewards[1] == 0
        assert summary["metrics"]["reward_mean"] == pytest.approx(np.mean(rewards))
    assert (tmp_path / "plots" / "fig7_sa.png").stat().st_size > 0
