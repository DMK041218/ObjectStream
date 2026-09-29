# ObjectStream

ObjectStream is a **training-free, object-centric visual memory** for streaming
video understanding with Qwen2.5-VL. It groups visual tokens into persistent
objects, updates object states over time, and retains compact evidence around
important temporal changes. No fine-tuning, detector, segmenter, or captioning
model is required.

This repository contains the complete Qwen2.5-VL integration, installable
patched model packages, lightweight inference entry points, and the evaluation
code used for OVO-Bench and StreamingBench.

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

PYTHON_BIN=python3.11 bash scripts/install_environment.sh
source .venv/bin/activate
```

The script creates `.venv`, installs the CUDA 12.4 build of PyTorch, installs
the two patched Qwen2.5-VL packages in editable mode, and validates the final
environment. To skip FlashAttention and use eager attention instead:

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

## Repository Layout

```text
objectstream/                 # public configuration and inference wrapper
models/
  qwen2-5-vl/                 # installable ObjectStream-patched Qwen2.5-VL
  qwen-vl-utils/              # installable image/video preprocessing helpers
ovo_ablation_work/            # OVO-Bench and memory-construction ablations
evaluation/streamingbench/    # StreamingBench evaluation
evaluation/ovobench/          # OVO-Bench scoring utility
requirements/                 # pinned runtime, evaluation, and development deps
scripts/                      # installation, validation, and inference entry points
tests/                        # training-free memory unit tests
```

## Validation

```bash
python scripts/check_environment.py
python scripts/run_qwen25vl_objectstream.py \
  --video_path dummy.mp4 \
  --question "Dry run" \
  --dry_run
python -m pytest tests
```

The dry run does not access model weights, datasets, videos, or GPUs. The unit
tests exercise the standalone object-memory component.

## Notes

- The released implementation targets **Qwen2.5-VL**.
- ObjectStream changes inference-time visual-memory construction only; model
  parameters are not updated.
- Model checkpoints and benchmark datasets are not stored in this repository.
- Results should be compared with identical frame sampling, resolution,
  generation length, and attention backend.
