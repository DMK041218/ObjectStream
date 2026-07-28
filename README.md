# ObjectStream

ObjectStream is a training-free visual memory module for long-video multimodal
inference. It compresses long visual token streams by organizing patch tokens
into persistent object-centric tracks and preserving compact evidence for each
track.

The repository contains the core ObjectStream implementation and lightweight
evaluation entry points used to integrate it with Qwen2.5-VL-style video
generation workflows.

## Core Idea

ObjectStream keeps long videos tractable by combining three operations:

- **Object-centric grouping:** spatially coherent visual tokens are grouped into
  soft object candidates in the model's visual-token space.
- **Persistent memory:** candidates are matched across time with an EMA-updated
  track feature, then ranked by persistence and saliency.
- **Temporal burst retention:** raw tokens around high-change moments are kept
  alongside compact object memory so abrupt actions or state changes are not
  over-compressed.

The method operates on visual tokens and does not require external object
detectors, segmentation models, finetuning, or additional captioning models.

## Repository Layout

```text
objectstream/
  config.py                 # configuration dataclasses
  entity_memory.py          # standalone object-memory bank
  qwen25vl_runner.py        # minimal Qwen2.5-VL ObjectStream runner

ovo_ablation_work/
  models/
    entitymem.py            # full ObjectStream / temporal burst implementation
    modeling_qwen2_5_vl_objectstream.py
    external_entity_frontends.py
  evaluation/
    ovobench.py
    ovobench_efficiency_qwen.py

evaluation/
  streamingbench/
    streamingbench.py
  llava_onevision_entitymem.py
  qwen3vl_entitymem.py

scripts/
  run_qwen25vl_objectstream.py
  prepare_stream_manifest.py

tests/
  test_entity_memory.py
```

## Basic Usage

Dry-run the lightweight runner without loading model weights:

```bash
python scripts/run_qwen25vl_objectstream.py \
  --video_path path/to/video.mp4 \
  --question "What happens in the video?" \
  --dry_run
```

Run with a Qwen2.5-VL-compatible ObjectStream backend:

```bash
python scripts/run_qwen25vl_objectstream.py \
  --video_path path/to/video.mp4 \
  --question "What happens in the video?" \
  --model_path Qwen/Qwen2.5-VL-7B-Instruct \
  --use_objectstream \
  --short_frames 8 \
  --medium_frames 64 \
  --max_entities 64 \
  --tokens_per_entity 8
```

## Important Parameters

- `short_frames`: recent frames kept with minimal compression.
- `medium_frames`: intermediate temporal budget before long-range compression.
- `max_entities`: maximum number of retained object tracks.
- `tokens_per_entity`: retained token budget per object track.
- `entity_match_threshold`: cross-frame object matching threshold.
- `ema_decay`: exponential moving average decay for persistent object features.
- `saliency_quantile`: saliency cutoff for candidate token grouping.

Temporal burst options are exposed in the benchmark entry points as
`entity_temporal_burst_*` flags.

## Tests

```bash
python -m pytest tests
```

The standalone tests cover the object-memory component and do not require video
datasets or model checkpoints.
