# SA-NNO-DRL

Reproduction of Fan et al., *Multi-UAV reconnaissance mission planning via deep reinforcement learning with simulated annealing*, Swarm and Evolutionary Computation 93 (2025), 101858. DOI: `10.1016/j.swevo.2025.101858`.

This repository builds on the authors' `mingfan321/SA-NNO-DRL` code, with runtime repairs, explicit padding masks, route validation, structured result export and route plotting.

## Reproduction status

The four experiments were completed on November 2, 2026. Each case uses a separately self-trained model, the final `epoch-9.pt` checkpoint, five UAVs and 30 test instances.

| Case | Training subproblem size | Test targets | Reward mean +/- sample SD | Time in seconds mean +/- sample SD |
| --- | --- | --- | --- | --- |
| T100U5C | 20 | 100 | 98.1667 +/- 1.2888 | 0.1698 +/- 0.0637 |
| T100U5U | 20 | 100 | 50.4427 +/- 3.5040 | 0.1702 +/- 0.0594 |
| T200U5C | 40 | 200 | 170.8000 +/- 3.9427 | 0.2639 +/- 0.0643 |
| T200U5U | 40 | 200 | 90.7850 +/- 4.4567 | 0.2538 +/- 0.0604 |

`C` means constant target prizes of 1; `U` means discrete uniform prizes from 0.01 to 1.00. Each UAV has a roundtrip length budget of 2.0. Saved results for all 120 instances and 600 UAV routes passed feasibility and reward consistency checks. Training curves and route figures have also been produced.

The scope is the SA-NNO-DRL method on these four cases. It excludes baseline comparisons, ablations and multiple training seeds, and does not establish optimality or reproduce every table in the paper. Standard deviations are across test instances. Time covers SA and route validation, including the first instance's cold startup, and excludes loading, export and plotting.

See [training-and-evaluation-records.md](doc/training-and-evaluation-records.md) for the configurations, commands, result definitions and analysis.

## Environment

The formal inference runs used Python 3.11.15, NumPy 2.4.6, PyTorch 2.14.1+cu130 and CUDA 13.0 on an NVIDIA GeForce RTX 5080. Driver 580.159.03 was recorded during the earlier environment check.

Use uv to reproduce the environment. The CUDA and CPU PyTorch extras are mutually exclusive:

```bash
uv sync --locked --extra cu130
uv run --extra cu130 python scripts/verify_environment.py --require-cuda
```

For a CPU-only logic environment:

```bash
uv sync --locked --extra cpu
uv run --extra cpu pytest -q
```

## Main flow

Run a small GPU smoke training job:

```bash
uv run --extra cu130 python run.py \
  --no_tensorboard --no_progress_bar \
  --graph_size 20 --batch_size 2 --epoch_size 2 \
  --val_size 2 --eval_batch_size 2 --n_epochs 1 \
  --checkpoint_epochs 1 --output_dir /tmp/sa-nno-smoke
```

Evaluate or resume from a checkpoint with `--load_path` or `--resume`. The checkpoint includes a format version, model/optimizer/scheduler states, RNG states and a configuration snapshot. Legacy checkpoints containing Python objects require the explicit `--allow_legacy_checkpoint` flag.

The four final self-trained checkpoints are stored in `weights/<case>/epoch-9.pt`, each with its paired `args.json`. The case directories are `T100U5C`, `T100U5U`, `T200U5C` and `T200U5U`.

Run SA inference with the T200U5C checkpoint:

```bash
uv run --extra cu130 python SA-NNO-DRL.py \
  --model weights/T200U5C/epoch-9.pt \
  --dataset op_data/const/op_const200_len2_seed12345.pkl \
  --num-samples 30 --uav-num 5 --device cuda:0 \
  --seed 1234 --weight-source self-trained \
  --output-dir results/sa
```

Keep `args.json` beside the checkpoint. When `--model` is a directory, the highest numbered `epoch-N.pt` is selected and its actual path is recorded; the formal experiments explicitly selected `epoch-9.pt`. For T200U5U, use `weights/T200U5U/epoch-9.pt` and `op_data/unif/op_unif200_len2_seed12345.pkl`.

`--graph_size` is the training subproblem size: 20 for T100 and 40 for T200. The training generator splits a mother instance with `graph_size * uav_num` targets into angular sectors. SA inference takes the full 100- or 200-target instance and allocates targets among UAVs. The four formal cases use their corresponding trained models.

## Saved inference results

Every inference run automatically creates a unique directory under `--output-dir` (default `results/sa`):

```text
results/sa/<dataset>_<UTC timestamp>_<unique suffix>/
  instances.csv
  routes.json
  summary.json
  model_args.json
```

| File | Contents |
| --- | --- |
| `instances.csv` | Dataset, zero-based instance ID, resolved checkpoint, inference seed, declared weight source, target/UAV counts, maximum route length, reward, seconds, visited target count and total route length |
| `routes.json` | Instance geometry, prizes, hub and each UAV's assigned target IDs, visited order, roundtrip length and collected reward |
| `summary.json` | Run status, requested/completed counts, mean and sample standard deviation of rewards and times, options, invocation, file hashes and runtime environment |
| `model_args.json` | Exact copy of the checkpoint's sibling `args.json` |

Target IDs are zero-based indices into each instance's `loc` and `prize`. The hub is stored separately: each route means `hub -> visited_target_ids -> hub`; an empty sequence means no flight. Assigned targets may include targets the UAV does not visit. The reward is positive collected prize, and equals the number of visited targets for const data.

Results are saved after each validated instance. `summary.json` reports `running`, `completed`, `interrupted` or `failed`; abrupt process termination can leave `running` with partial results. Each file is replaced atomically, with the summary written last. Use completed runs for tables. Requested counts can exceed the dataset length; the summary records the actual count. Existing repository datasets have 30 instances each.

The standard deviation is across instances (`ddof=1`), with `null` for fewer than two instances. Time measures SA plus allocation validation, synchronized on CUDA, and excludes model/data loading, export and plotting. It includes cold startup in the first instance. There is no separate warmup.

`--seed` seeds NumPy and PyTorch; omission records `null` and leaves the run unseeded. A fixed seed is useful for repeated runs in the same environment, but does not guarantee identical results across hardware or library versions. `--weight-source` records the declared provenance (`self-trained`, `external`, or default `unspecified`). File hashes identify the dataset, checkpoint and original model configuration. Git revision/dirty status describe the **inference** checkout; they do not establish the training code version.

Optional Fig. 7-style plotting remains available:

```bash
# Append these options to an inference command for a small set of instances:
--plot-dir results/fig7 --plot-name fig7_sa --plot-format png
```

Plotting puts all evaluated instances in one row, so use a small `--num-samples` for readable figures. Structured results are saved with or without plotting.

The completed route figures show instances 0, 1 and 2 from each formal case, using the saved routes. They follow the SA row's visual style in Fig. 7, with a fixed central hub and no OR-Tools comparison row. Adding plotting options to an inference command performs a new inference run; it does not reload the saved formal routes.
