# ObjectStream · Qwen3.5-9B

[📄 Paper](https://arxiv.org/abs/2607.28312) · [Project overview](../README.md)

Training-free object-centric visual memory for Qwen3.5-9B, using its native
visual encoder, processor, and hybrid decoder. This directory provides an
independent implementation and does not require the Qwen2.5-VL packages.

All commands below run from this directory. For the other backbone, see
[Qwen2.5-VL-7B](../qwen2.5-vl-7b/).

## Installation

Requirements: Linux, Python 3.12, an NVIDIA GPU with a CUDA 12.x driver, and
FFmpeg. Use a separate environment from the Qwen2.5-VL implementation.

```bash
git clone https://github.com/DMK041218/ObjectStream.git
cd ObjectStream/qwen3.5-9b

PYTHON_BIN=python3.12 bash scripts/install_environment.sh
source .venv/bin/activate
```

The installer uses PyTorch 2.6.0 / torchvision 0.21.0 with CUDA 12.4 and
Transformers 5.18.0. FlashAttention is not required. See
[Installation Guide](docs/INSTALLATION.md) for manual setup.

## Quick Start

Check the entry point without loading a model or video:

```bash
python scripts/run_qwen35_objectstream.py \
  --video_path example.mp4 --question "What happens in the video?" \
  --timestamp 10 --dry_run
```

Run inference:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_qwen35_objectstream.py \
  --model_path Qwen/Qwen3.5-9B \
  --video_path /path/to/example.mp4 \
  --question "What happens in the video?" --timestamp 10
```

`--timestamp` is the last observable source-video time in seconds.
`--model_path` accepts a Hugging Face model identifier or a local checkpoint
directory; add `--local_files_only` for local-only loading. Use `--mode native`
for a reference without ObjectStream. By default, the runner expects one
visible GPU; set `--cards 2` with exactly two visible GPUs for model splitting.

## Continuous Queries

```python
from objectstream.qwen35_runner import Qwen35ObjectStream

runner = Qwen35ObjectStream("Qwen/Qwen3.5-9B")
first = runner.answer("video.mp4", "What is visible?", timestamp=10)
second = runner.answer("video.mp4", "What changed?", timestamp=20)
runner.reset()  # Start an independent video or episode.
```

The same runner carries compressed visual features between chronological
queries. For complete-video segment ingestion, memory semantics, and sampling
limits, see [Implementation Notes](docs/IMPLEMENTATION.md).

## Configuration

| Setting | Default |
|---|---:|
| Sampling rate | 1 fps |
| Maximum new sampled frames per update | 256 |
| Recent frames | 8 |
| Medium-frame budget | 256 |
| Maximum entities / tokens per entity | 256 / 32 |
| Saliency quantile / matching threshold | 0.60 / 0.72 |
| Temporal events / radius / tokens per event frame | 32 / 1 / 2 |
| Generation | BF16, SDPA, non-thinking, greedy, up to 128 tokens |

The [configuration summary](configs/qwen35_9b_objectstream.json) documents
these defaults; it is not a CLI configuration-file loader.

## Evaluation

OVO-Bench real-time / backward and StreamingBench manifest preparation,
inference, and scoring are documented in the
[Evaluation Guide](docs/EVALUATION.md).

## Checks

```bash
python scripts/check_environment.py
python -m pytest tests -q
```

## Directory Layout

```text
objectstream/       # Native Qwen3.5 adapter, visual memory, and public runner
scripts/            # Installation, inference, manifests, and scoring
configs/            # Default configuration summary
requirements/       # Pinned Qwen3.5 environment
docs/               # Installation, evaluation, and implementation notes
tests/              # Import and memory-contract tests
```
