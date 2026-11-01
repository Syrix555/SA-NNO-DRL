# SA-NNO-DRL

Code for Swarm and Evolutionary Computation Paper: Multi-UAV Reconnaissance Mission Planning via Deep Reinforcement Learning with Simulated Annealing

## Environment

The maintained runtime is Python 3.11 with PyTorch 2.14.1 and CUDA 13.0. The validated GPU target is an NVIDIA RTX 5080 (compute capability 12.0) with driver 580.159.03.

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

Run SA inference with a trained checkpoint:

```bash
uv run --extra cu130 python SA-NNO-DRL.py \
  --model outputs/paper/op_40/t40_const_full_20261028T220058/epoch-9.pt \
  --dataset op_data/const/op_const40_len2_seed12345.pkl \
  --num-samples 30 --uav-num 5 --device cuda:0 \
  --seed 1234 --weight-source self-trained \
  --output-dir results/sa
```

The checkpoint path above is from the user's completed GPU training run; weights are kept on that machine. When `--model` is a directory, the highest numbered `epoch-N.pt` is selected and its actual path is recorded. Keep `args.json` beside the checkpoint. For the unif run, use `t40_unif_full_20261029T030049/epoch-9.pt` and `op_data/unif/op_unif40_len2_seed12345.pkl`.

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

The target environment has been validated for CUDA startup, GPU training, checkpoint evaluation/resume and SA inference. The user reports completed const/unif training with checkpoints, CSV curves and TensorBoard events; the artifact inventory and remaining provenance gaps are recorded in [training-and-evaluation-records.md](doc/training-and-evaluation-records.md). Result export has been tested locally with a small CPU inference run; the updated exporter still needs to be exercised on the target GPU. Baseline comparisons, Gap and the full paper experiment matrix remain separate work.
