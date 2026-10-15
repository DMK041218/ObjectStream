# Evaluation Guide

This page describes Qwen2.5. For the separate native Qwen3.5-9B environment,
chronological manifests and scoring, see [Qwen3.5](QWEN35.md).

## Archived e64/t8/burst profile (Qwen2.5)

The archived launcher used recent8, medium64, entity64/token8,
saliency.60, match.72, burst32/radius1/token2,1fps, max256 frames,
max401408 pixels and time_window_size256. Minimal README/demo defaults differ.
For real-time perception only, explicitly select the six tasks:

```bash
CUDA_VISIBLE_DEVICES=0 python ovo_ablation_work/evaluation/ovobench.py \
  --ckpt_path /path/to/Qwen2.5-VL-7B-Instruct \
  --task_json /path/to/OVO-Bench/ovo_bench_new.json --video_dir /path/to/OVO-Bench \
  --result_dir outputs/ovo_qwen25_e64_t8_burst \
  --task STU OJR ATR ACR OCR FPD --fps 1 --max_pixels 401408 --max_frames 256 \
  --time_window_size 256 --use_entitymem --short_frames 8 --medium_frames 64 \
  --entity_max_entities 64 --entity_tokens_per_entity 8 --entity_saliency_quantile 0.60 \
  --entity_match_threshold 0.72 --entity_temporal_burst_selection \
  --entity_temporal_burst_max_events 32 --entity_temporal_burst_radius 1 \
  --entity_temporal_burst_tokens_per_frame 2
```

The original evaluator expects the benchmark's `chunked_videos` layout under
`--video_dir`; `src_videos` alone is not a substitute for those prepared clips.
This publishes the archived profile, not a claim of a fresh73.3 rerun after
the packaging fix. The evaluator uses FlashAttention2; eager demo execution
is an installation check, not the identical speed/reproduction configuration.

Install the evaluation dependencies before running benchmarks:

```bash
source .venv/bin/activate
python -m pip install -r requirements/evaluation.txt
```

The benchmark datasets and Qwen2.5-VL weights are prepared separately and are
not committed to this repository. Use absolute paths in the commands below.

## OVO-Bench

Single GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python ovo_ablation_work/evaluation/ovobench.py \
  --ckpt_path /path/to/Qwen2.5-VL-7B-Instruct \
  --task_json /path/to/OVO-Bench/ovo_bench_new.json \
  --video_dir /path/to/OVO-Bench/videos \
  --result_dir outputs/ovobench_objectstream \
  --use_objectstream \
  --short_frames 8 \
  --medium_frames 64 \
  --entity_max_entities 64 \
  --entity_tokens_per_entity 8
```

Multi-GPU example:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python ovo_ablation_work/evaluation/ovobench.py \
  --ckpt_path /path/to/Qwen2.5-VL-7B-Instruct \
  --task_json /path/to/OVO-Bench/ovo_bench_new.json \
  --video_dir /path/to/OVO-Bench/videos \
  --result_dir outputs/ovobench_objectstream \
  --use_objectstream \
  --multi_gpu \
  --num_gpus 4 \
  --gpu_ids 0 1 2 3
```

The evaluator writes JSONL predictions and invokes
`evaluation/ovobench/score.py` after generation.

## StreamingBench

Single GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python evaluation/streamingbench/streamingbench.py \
  --ckpt_path /path/to/Qwen2.5-VL-7B-Instruct \
  --task_csv /path/to/StreamingBench/questions.csv \
  --video_dir /path/to/StreamingBench/videos \
  --result_dir outputs/streamingbench_objectstream \
  --use_objectstream \
  --short_frames 8 \
  --medium_frames 64 \
  --entity_max_entities 64 \
  --entity_tokens_per_entity 8
```

Multi-GPU example:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python evaluation/streamingbench/streamingbench.py \
  --ckpt_path /path/to/Qwen2.5-VL-7B-Instruct \
  --task_csv /path/to/StreamingBench/questions.csv \
  --video_dir /path/to/StreamingBench/videos \
  --result_dir outputs/streamingbench_objectstream \
  --use_objectstream \
  --multi_gpu \
  --num_gpus 4 \
  --gpu_ids 0 1 2 3
```

Score an existing result directory:

```bash
python evaluation/streamingbench/score.py \
  --result_dir outputs/streamingbench_objectstream
```

## Baseline comparison

For a controlled baseline, repeat the same command without
`--use_objectstream`. Keep the checkpoint, frame sampling rate, pixel budget,
generation length, and attention backend unchanged.

## Generation-latency measurement

The latency runner reports generation time after the video has already been
prepared, so video decoding and preprocessing are not counted as answer
latency:

```bash
CUDA_VISIBLE_DEVICES=0 python evaluation_latency_generation_microbench.py \
  --ckpt_path /path/to/Qwen2.5-VL-7B-Instruct \
  --task_json /path/to/OVO-Bench/ovo_bench_new.json \
  --video_dir /path/to/OVO-Bench/videos \
  --out_dir outputs/latency_objectstream \
  --method entitymem \
  --max_new_tokens 512
```

Report the same maximum generation length for all methods. In addition to
wall-clock answer time, keep TTFT, decode throughput, generated-token count,
and truncation rate so faster decoding is not conflated with shorter outputs.
