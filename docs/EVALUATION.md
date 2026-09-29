# Evaluation Guide

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
