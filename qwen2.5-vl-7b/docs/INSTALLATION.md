# Installation Guide

This guide reproduces the Qwen2.5-VL ObjectStream environment used by the
released code. Commands assume Linux, Python 3.11, and an NVIDIA CUDA 12.x
driver.

## 1. System packages

On Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential \
  ffmpeg \
  git \
  ninja-build \
  python3.11 \
  python3.11-dev \
  python3.11-venv
```

The NVIDIA driver must already be installed. A separate system-wide CUDA
toolkit is not needed for the prebuilt PyTorch wheel, but compiling
FlashAttention locally requires a compatible CUDA toolkit and `nvcc`.

## 2. Recommended automated setup

From the `qwen2.5-vl-7b/` directory:

```bash
PYTHON_BIN=python3.11 \
VENV_DIR=.venv \
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 \
bash scripts/install_environment.sh
```

Activate and verify:

```bash
source .venv/bin/activate
python scripts/check_environment.py
```

Environment variables supported by the installer:

| Variable | Default | Description |
|---|---|---|
| `PYTHON_BIN` | `python3.11` | Python executable used to create the environment |
| `VENV_DIR` | `.venv` | Virtual-environment directory |
| `TORCH_INDEX_URL` | PyTorch CUDA 12.4 index | PyTorch wheel index |
| `INSTALL_FLASH_ATTN` | `1` | Set to `0` to skip FlashAttention |

## 3. Manual setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate

python -m pip install pip==26.2.1 setuptools==75.8.2 wheel==0.45.1
python -m pip install \
  --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.6.0 torchvision==0.21.0
python -m pip install -r requirements/runtime.txt
python scripts/repair_decord_wheel.py
python -m pip install -e models/qwen-vl-utils
python -m pip install -e models/qwen2-5-vl
python -m pip install -e .
```

Install the tested FlashAttention version after PyTorch:

```bash
python -m pip install flash-attn==2.7.4.post1 --no-build-isolation
```

If FlashAttention cannot be installed, ObjectStream can run with eager
attention:

```bash
python scripts/run_qwen25vl_objectstream.py \
  --video_path path/to/video.mp4 \
  --question "Describe the video." \
  --attn_implementation eager
```

## 4. Evaluation extras

Install benchmark and analysis dependencies only when needed:

```bash
python -m pip install -r requirements/evaluation.txt
```

For development and unit tests:

```bash
python -m pip install -r requirements/dev.txt
python -m pytest tests
```

## 5. Model checkpoint handling

The runner accepts either a model identifier or a local directory:

```bash
# Cached or remotely resolved model identifier
--model_path Qwen/Qwen2.5-VL-7B-Instruct

# Existing local checkpoint
--model_path /absolute/path/to/Qwen2.5-VL-7B-Instruct --local_files_only
```

No converted ObjectStream checkpoint is required. The patched model class
loads the standard Qwen2.5-VL weights and applies ObjectStream during inference.

## 6. Sanity checks

The environment checker verifies Python, PyTorch, Transformers, CUDA, and both
editable patched packages:

```bash
python scripts/check_environment.py
```

The CLI can be checked without any weights or media:

```bash
python scripts/run_qwen25vl_objectstream.py \
  --video_path dummy.mp4 \
  --question "Dry run" \
  --dry_run
```

## 7. Troubleshooting

The installer normalizes Decord 0.6's legacy Linux wheel tag to match the
official wheel filename, updating its RECORD checksum as well. Decoder code
and binaries are unchanged; this allows current pip compatibility checks.

### `flash_attn` import or build failure

Confirm that PyTorch is installed before FlashAttention and that the CUDA
toolkit used to compile FlashAttention is compatible with the PyTorch CUDA
runtime. Otherwise use `--attn_implementation eager`.

### FFmpeg or video decoding failure

Confirm both commands succeed:

```bash
ffmpeg -version
python -c "import av, cv2; print(av.__version__, cv2.__version__)"
```

### Patched package not found

Reinstall both editable packages from the `qwen2.5-vl-7b/` directory:

```bash
python -m pip install -e models/qwen-vl-utils
python -m pip install -e models/qwen2-5-vl
```

### Out-of-memory errors

Reduce `--max_pixels`, lower the video `--fps`, or use the 3B checkpoint. Keep
the baseline and ObjectStream settings identical when reporting comparisons.
