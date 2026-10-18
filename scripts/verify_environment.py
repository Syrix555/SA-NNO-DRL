#!/usr/bin/env python3
"""Verify the pinned Python/PyTorch/CUDA runtime on the target machine."""

import argparse
import importlib
import platform
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--require-cuda', action='store_true')
    args = parser.parse_args()

    print(f"Python: {platform.python_version()} ({sys.executable})")
    if sys.version_info[:2] != (3, 11):
        raise SystemExit('Expected Python 3.11')

    torch = importlib.import_module('torch')
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA runtime: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if not torch.__version__.startswith('2.14.1'):
        raise SystemExit('Expected PyTorch 2.14.1')

    if args.require_cuda:
        if torch.version.cuda != '13.0' or not torch.cuda.is_available():
            raise SystemExit('Expected a CUDA 13.0 PyTorch build with CUDA available')
        name = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)
        print(f"GPU: {name}")
        print(f"GPU capability: {capability}")
        if capability != (12, 0):
            raise SystemExit(f'Expected RTX 5080 capability (12, 0), got {capability}')
        x = torch.randn(128, 128, device='cuda')
        result = (x @ x).mean().item()
        torch.cuda.synchronize()
        print(f"GPU matmul: {result}")

    for package in ('numpy', 'scipy', 'pandas', 'matplotlib', 'tqdm', 'tensorboard'):
        module = importlib.import_module(package)
        print(f"{package}: {getattr(module, '__version__', 'available')}")

    try:
        driver = subprocess.run(
            ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
            check=False, capture_output=True, text=True,
        )
        if driver.returncode == 0:
            print(f"NVIDIA driver: {driver.stdout.strip()}")
    except OSError:
        pass


if __name__ == '__main__':
    main()
