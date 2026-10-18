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
uv run --extra cpu pytest tests/test_stage1_data.py -q
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
  --model /path/to/epoch-1.pt \
  --dataset op_data/const/op_const40_len2_seed12345.pkl \
  --num-samples 1 --uav-num 5 --device cuda:0
```

The target environment has been validated for CUDA startup, GPU training, checkpoint evaluation and resume, and one-sample five-UAV SA inference. Full paper-scale training and metrics are not part of the smoke validation.
