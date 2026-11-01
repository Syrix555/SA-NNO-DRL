"""Persist validated SA routes and the evidence needed to compare runs."""

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile

import numpy as np


CSV_FIELDS = (
    "dataset", "instance_id", "model", "seed", "weight_source", "target_count",
    "uav_num", "max_length", "reward", "duration_seconds",
    "visited_target_count", "total_route_length",
)


def file_metadata(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(path),
        "sha256": digest.hexdigest(),
        "size_bytes": path.stat().st_size,
    }


def runtime_metadata(device):
    import torch

    repo = Path(__file__).resolve().parent
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo, stderr=subprocess.DEVNULL,
            text=True,
        ).strip())
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
        ),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "inference_git_revision": revision,
        "inference_git_dirty": dirty,
    }


def instance_result(instance_id, batch, reward, duration, assignments, lines):
    """Translate local decoder indices to zero-based dataset target IDs."""
    task = np.asarray(batch["loc"], dtype=np.float32)
    depot = np.asarray(batch["depot"], dtype=np.float32).reshape(2)
    prize = np.asarray(batch["prize"], dtype=np.float32)
    if len(assignments) != len(lines):
        raise ValueError("Each UAV assignment must have one route")
    routes = []
    for uav_id, (assignment, line) in enumerate(zip(assignments, lines)):
        assignment = np.asarray(assignment, dtype=np.int64)
        line = np.asarray(line, dtype=np.int64)
        visited = assignment[line]
        route = np.concatenate((depot[None], task[visited], depot[None]), axis=0)
        routes.append({
            "uav_id": uav_id,
            "assigned_target_ids": assignment.tolist(),
            "visited_target_ids": visited.tolist(),
            "reward": float(prize[visited].sum()),
            "route_length": float(np.linalg.norm(np.diff(route, axis=0), axis=1).sum()),
        })
    return {
        "instance_id": int(instance_id),
        "target_count": len(task),
        "uav_num": len(routes),
        "max_length": float(np.asarray(batch["max_length"])),
        "reward": float(reward),
        "duration_seconds": float(duration),
        "visited_target_count": sum(len(r["visited_target_ids"]) for r in routes),
        "total_route_length": sum(r["route_length"] for r in routes),
        "depot": depot.tolist(),
        "loc": task.tolist(),
        "prize": prize.tolist(),
        "routes": routes,
    }


class InferenceResults:
    """Use a unique directory; refresh snapshots after every validated instance."""

    def __init__(self, options, model_path, device):
        self.options = dict(options)
        self.records = []
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.metadata = {
            "dataset": file_metadata(options["dataset"]),
            "checkpoint": file_metadata(model_path),
            "model_args": file_metadata(Path(model_path).parent / "args.json"),
            "weight_source": options["weight_source"],
            "environment": runtime_metadata(device),
            "argv": [sys.executable, *sys.argv],
            "working_directory": str(Path.cwd()),
        }
        root = Path(options["output_dir"]).resolve()
        root.mkdir(parents=True, exist_ok=True)
        prefix = (
            Path(options["dataset"]).stem
            + datetime.now(timezone.utc).strftime("_%Y%m%dT%H%M%S_")
        )
        self.directory = Path(tempfile.mkdtemp(prefix=prefix, dir=root))
        shutil.copyfile(self.metadata["model_args"]["path"],
                        self.directory / "model_args.json")
        self.save("running")

    def add(self, record):
        if record["uav_num"] != self.options["uav_num"]:
            raise ValueError("Result UAV count differs from the requested count")
        if not np.isclose(
            sum(route["reward"] for route in record["routes"]),
            record["reward"], rtol=1e-5, atol=1e-5,
        ):
            raise ValueError("Exported route reward differs from reported reward")
        self.records.append(record)
        self.save("running")

    def _write_json(self, filename, value):
        temporary = self.directory / (filename + ".tmp")
        temporary.write_text(
            json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8",
        )
        temporary.replace(self.directory / filename)

    def save(self, status):
        common = {
            "dataset": self.metadata["dataset"]["path"],
            "model": self.metadata["checkpoint"]["path"],
            "seed": self.options["seed"],
            "weight_source": self.options["weight_source"],
        }
        temporary = self.directory / "instances.csv.tmp"
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for record in self.records:
                writer.writerow({key: (record | common)[key] for key in CSV_FIELDS})
        temporary.replace(self.directory / "instances.csv")
        self._write_json("routes.json", {
            "schema_version": 1,
            **common,
            "target_id_convention": "Zero-based indices into loc/prize; hub is separate.",
            "route_convention": "depot -> visited_target_ids -> depot; empty means no flight.",
            "instances": self.records,
        })
        metrics = {}
        for key in ("reward", "duration_seconds"):
            values = [record[key] for record in self.records]
            metrics[key + "_mean"] = float(np.mean(values)) if values else None
            metrics[key + "_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        self._write_json("summary.json", {
            "schema_version": 1,
            "status": status,
            "started_at_utc": self.started_at,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "requested_samples": self.options["num_samples"],
            "completed_samples": len(self.records),
            "std_definition": "Across instances, sample standard deviation (ddof=1); null for n<2.",
            "timing_definition": (
                "Per-instance SA and allocation validation, synchronized on CUDA; "
                "excludes loading, export and plotting; includes first-instance cold startup."
            ),
            "metrics": metrics,
            "options": self.options,
            "metadata": self.metadata,
        })
