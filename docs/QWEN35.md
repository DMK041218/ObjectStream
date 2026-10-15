# Qwen3.5-9B

Training-free integration with the unmodified native Transformers Qwen3.5
visual encoder, processor and hybrid decoder. This is NOT the patched Qwen2.5
decoder with a different checkpoint. It uses BF16/SDPA, non-thinking greedy
generation128, seed1234 and explicit native temporal/height/width positions.

## Install in a separate environment

```bash
PYTHON_BIN=python3.12 bash scripts/install_qwen35_environment.sh
source .venv-qwen35/bin/activate
python scripts/check_environment.py --backbone qwen35
OBJECTSTREAM_TEST_BACKBONE=qwen35 python -m pytest tests -q
```

No FlashAttention or custom linear-attention kernel is required for correctness;
the tested runtime can use slower PyTorch fallback kernels. Install system
FFmpeg for video decoding. Model weights/datasets are downloaded separately.
The core package does not install or import the legacy Qwen2.5 packages here.

## One query

```bash
python scripts/run_qwen35_objectstream.py --video_path example.mp4 \
  --question 'What happens in the video?' --timestamp 10 --dry_run

CUDA_VISIBLE_DEVICES=0 python scripts/run_qwen35_objectstream.py \
  --model_path Qwen/Qwen3.5-9B --video_path /path/to/example.mp4 \
  --question 'What happens in the video?' --timestamp 10
```

`timestamp` is the last observable source-video time in seconds, not a wall
clock. A local checkpoint directory plus `--local_files_only` is supported.
The CLI requires an explicitly matching visible-device count (`--cards`).
`--mode native` disables ObjectStream for a native-model reference.
`--variant full` is the default; component variants are also available.

## Continuous queries and video ingestion

```python
from objectstream.qwen35_runner import Qwen35ObjectStream

runner = Qwen35ObjectStream('/path/to/Qwen3.5-9B', local_files_only=True)
first = runner.answer('/path/to/video.mp4', 'What is visible?', timestamp=10)
second = runner.answer('/path/to/video.mp4', 'What changed?', timestamp=20)
runner.reset()  # New independent video/episode.
```

The same runner instance carries compressed visual features and source-time
geometry, not previous generated answers or an LLM KV cache. Equal-time queries
reuse identical prepared visual memory. Earlier-than-cached queries are rejected.
Entity states are reconstructed during compression of retained features;
this is not persistent entity-ID tracking across separate calls.

The default reader matches the evaluated profile:1fps, end-anchored, at most
256 **new** sampled frames per update. A large gap between updates therefore
does NOT ingest all intervening footage. For complete-video offline evaluation
or an external live frame source, call `runner.ingest_frames(video_id, frames,
source_indices, source_pts, source_fps)` repeatedly with bounded chronological
TCHW raw RGB frame batches, then `answer()` at the final observed source PTS.
Use an even batch size>=4; any temporal padding must duplicate already-observed
past frames only. Ingestion performs no answer generation. No fixed17.5% budget
is enabled in this release; maxima do not imply a fixed actual retention ratio.

## OVO Realtime / Backward and StreamingBench

```bash
python scripts/prepare_qwen35_manifest.py --suite ovo-realtime \
  --annotations /path/to/ovo_bench_new.json --video_dir /path/to/OVO-Bench/src_videos \
  --output outputs/ovo_rt_manifest.json

CUDA_VISIBLE_DEVICES=0 python scripts/run_qwen35_objectstream.py \
  --model_path /path/to/Qwen3.5-9B --local_files_only \
  --manifest outputs/ovo_rt_manifest.json --output outputs/ovo_rt_answers.jsonl

python scripts/score_qwen35.py --manifest outputs/ovo_rt_manifest.json \
  --answers outputs/ovo_rt_answers.jsonl
```

Use `--suite ovo-backward` for EPM/ASI/HLD. For StreamingBench use
`--suite streaming`, the original question CSV and its `videos/` directory.
The builder preserves the evaluated question templates and includes all IDs
by default. Historical2498-question StreamingBench runs excluded exactly
`Real-Time Visual Understanding_sample_332_4` and `_332_5`; to reproduce THAT
scope, explicitly pass both full IDs via repeated `--exclude_id`, and disclose
them. They are not silently removed by this release.

Manifest rows contain `id,video_path,question,timestamp` and optional label/task
metadata; the runner passes only video, prompt and timestamp to inference.
It sorts whole videos chronologically, resets between unrelated videos, keeps
raw answers and telemetry, and refuses to overwrite an existing output file.
Use one model worker per process (sampler instrumentation is not thread-safe).
External sharding must keep all questions from one video in the same worker;
use a disjoint manifest per worker. There is no hardcoded GPU allocation.

The scorer requires complete unique manifest coverage and reports historical
task-macro scores AND strict leading-option scores separately. The historical
OVO rule checks label occurrence in the raw response; it can overcount verbose
answers. Do not compare scores from different parsing policies as equivalent.
Forward OVO episodes require the official nested scorer and are not covered by
this portable manifest builder.

Default budgets are in `configs/qwen35_9b_objectstream.json` (descriptive, not a
config-file loader): recent8, medium256, entity256/token32, saliency.60,
match.72, OCTR32 events/radius1/token2. The new17.5% budget sweep is deliberately
not included in or substituted for these validated defaults.
