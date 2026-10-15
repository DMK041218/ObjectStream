# ObjectStream · Qwen2.5-VL-7B

Training-free object-centric visual memory for Qwen2.5-VL-7B. This directory
contains the modified model, processor, video utilities, inference runner, and
OVO-Bench / StreamingBench evaluation scripts.

All commands below run from this directory. For the other backbone, see
[Qwen3.5-9B](../qwen3.5-9b/).

## Requirements

- Linux (Ubuntu 22.04 is recommended)
- Python 3.11
- NVIDIA GPU with CUDA 12.x
- PyTorch 2.6.0 and torchvision 0.21.0
- FFmpeg available on `PATH`
- 24 GB or more GPU memory for Qwen2.5-VL-7B; 48 GB is recommended for long videos

The tested package versions are pinned under [`requirements/`](requirements).
ObjectStream itself is training-free; DeepSpeed and training libraries are not
required for inference.

## Installation

Clone the repository and run the environment installer:

```bash
git clone https://github.com/DMK041218/ObjectStream.git
cd ObjectStream
cd qwen2.5-vl-7b

PYTHON_BIN=python3.11 bash scripts/install_environment.sh
source .venv/bin/activate
```

The script creates `.venv`, installs PyTorch with CUDA 12.4 and both local
packages (`qwen2_5_vl_objectstream` and `qwen_vl_utils_objectstream`), and checks
the environment. To skip FlashAttention and use eager attention instead:

```bash
INSTALL_FLASH_ATTN=0 bash scripts/install_environment.sh
source .venv/bin/activate
python scripts/check_environment.py
```

For manual installation, system-package notes, other CUDA configurations, and
troubleshooting, see [Installation Guide](docs/INSTALLATION.md).

## Quick Start

First verify the command path without loading a model or video:

```bash
python scripts/run_qwen25vl_objectstream.py \
  --video_path path/to/video.mp4 \
  --question "What happens in the video?" \
  --dry_run
```

Run ObjectStream with Qwen2.5-VL-7B:

```bash
python scripts/run_qwen25vl_objectstream.py \
  --video_path path/to/video.mp4 \
  --question "What happens in the video?" \
  --model_path Qwen/Qwen2.5-VL-7B-Instruct \
  --fps 1.0 \
  --short_frames 8 \
  --medium_frames 64 \
  --max_entities 64 \
  --tokens_per_entity 8 \
  --use_objectstream
```

`--model_path` accepts either a local Qwen2.5-VL checkpoint directory or a
Hugging Face model identifier. Add `--local_files_only` when the checkpoint is
already present locally. If FlashAttention was not installed, add
`--attn_implementation eager`.

Run the uncompressed Qwen2.5-VL baseline with the same entry point:

```bash
python scripts/run_qwen25vl_objectstream.py \
  --video_path path/to/video.mp4 \
  --question "What happens in the video?" \
  --model_path Qwen/Qwen2.5-VL-7B-Instruct \
  --no_objectstream
```

## Main Parameters

| Argument | Default | Meaning |
|---|---:|---|
| `--fps` | `1.0` | Video sampling rate passed to the Qwen processor |
| `--short_frames` | `8` | Recent frames retained at full visual-token resolution |
| `--medium_frames` | `64` | Intermediate temporal window before long-range compression |
| `--max_entities` | `64` | Maximum number of persistent object tracks |
| `--tokens_per_entity` | `8` | Visual-token budget retained per object track |
| `--entity_match_threshold` | `0.72` | Cross-frame cosine-similarity threshold for object matching |
| `--ema_decay` | `0.9` | EMA coefficient for persistent object features |
| `--saliency_quantile` | `0.75` | Saliency cutoff used to construct object candidates |
| `--max_pixels` | `200704` | Maximum pixels per sampled video frame |

The complete example configuration is in
[`configs/qwen25vl_objectstream.example.json`](configs/qwen25vl_objectstream.example.json).

## Evaluation

OVO-Bench, StreamingBench, latency measurement, multi-GPU launch examples, and
the corresponding scoring commands are documented in
[Evaluation Guide](docs/EVALUATION.md).

## Directory Layout

```text
objectstream/                 # public configuration and inference wrapper
models/
  qwen2-5-vl/                 # installable ObjectStream-patched Qwen2.5-VL
  qwen-vl-utils/              # installable image/video preprocessing helpers
evaluation/ovobench/          # OVO-Bench inference and scoring
evaluation/streamingbench/    # StreamingBench evaluation
requirements/                 # pinned runtime, evaluation, and development deps
scripts/                      # installation, validation, and inference entry points
tests/                        # training-free memory unit tests
```

## Checks

```bash
python scripts/check_environment.py
python scripts/run_qwen25vl_objectstream.py \
  --video_path dummy.mp4 \
  --question "Dry run" \
  --dry_run
python -m pytest tests
```

The dry run does not load model weights or videos. Tests cover the packaged
model and processor imports, command-line entry points, and memory helpers.

## Notes

- ObjectStream changes inference-time visual-memory construction only; model
  parameters are not updated.
- Memory is constructed across frames within each supplied video prefix;
  independent generation calls do not share a visual-memory cache. See
  [Implementation Notes](docs/IMPLEMENTATION.md).
- Results should be compared with identical frame sampling, resolution,
  generation length, and attention backend.
