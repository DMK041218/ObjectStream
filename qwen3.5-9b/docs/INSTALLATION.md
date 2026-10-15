# Installation Guide

Run all commands from `qwen3.5-9b/`. This implementation uses Python 3.12 and
Transformers 5.18.0; do not install the Qwen2.5-VL packages in this environment.
On Linux, install FFmpeg and ensure an NVIDIA CUDA 12.x driver is available.

## Automated setup

```bash
PYTHON_BIN=python3.12 bash scripts/install_environment.sh
source .venv/bin/activate
python scripts/check_environment.py
```

The installer accepts `PYTHON_BIN`, `VENV_DIR` (default `.venv`), and
`TORCH_INDEX_URL` (default PyTorch CUDA 12.4 index).

## Manual setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.6.0 torchvision==0.21.0
python -m pip install -r requirements/runtime.txt
python -m pip install -e .
python scripts/check_environment.py
python -m pytest tests -q
```

The native model runs with BF16 and SDPA. FlashAttention and custom
linear-attention kernels are optional; PyTorch fallback kernels are supported.
Model weights are loaded from a standard Qwen3.5-9B checkpoint, with no
ObjectStream-specific weight conversion or fine-tuning.

## Input-layout check

With a local checkpoint's configuration / processor files and a video:

```bash
CUDA_VISIBLE_DEVICES= python scripts/validate_qwen35_inputs.py \
  --model_path /path/to/Qwen3.5-9B --video_path /path/to/video.mp4 \
  --timestamp 10
```

This CPU check compares the reconstructed full-retention input IDs and visual
position axes with the native Qwen3.5 input layout without loading model weights.
