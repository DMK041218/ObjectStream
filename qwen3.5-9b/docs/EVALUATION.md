# Evaluation Guide

Run all commands from `qwen3.5-9b/` with its environment activated. Prepare
the model checkpoint and benchmark videos separately.

## OVO-Bench

Create a real-time perception manifest, run inference, and score:

```bash
python scripts/prepare_qwen35_manifest.py --suite ovo-realtime \
  --annotations /path/to/OVO-Bench/ovo_bench_new.json \
  --video_dir /path/to/OVO-Bench/src_videos \
  --output outputs/ovo_rt_manifest.json

CUDA_VISIBLE_DEVICES=0 python scripts/run_qwen35_objectstream.py \
  --model_path /path/to/Qwen3.5-9B --local_files_only \
  --manifest outputs/ovo_rt_manifest.json \
  --output outputs/ovo_rt_answers.jsonl

python scripts/score_qwen35.py --manifest outputs/ovo_rt_manifest.json \
  --answers outputs/ovo_rt_answers.jsonl
```

Real-time perception includes STU, OJR, ATR, ACR, OCR, and FPD. Use
`--suite ovo-backward` for EPM, ASI, and HLD. Forward episodes use the official
nested OVO scorer and are not supported by this manifest builder.

## StreamingBench

```bash
python scripts/prepare_qwen35_manifest.py --suite streaming \
  --annotations /path/to/StreamingBench/questions.csv \
  --video_dir /path/to/StreamingBench/videos \
  --output outputs/streaming_manifest.json

CUDA_VISIBLE_DEVICES=0 python scripts/run_qwen35_objectstream.py \
  --model_path /path/to/Qwen3.5-9B --local_files_only \
  --manifest outputs/streaming_manifest.json \
  --output outputs/streaming_answers.jsonl

python scripts/score_qwen35.py --manifest outputs/streaming_manifest.json \
  --answers outputs/streaming_answers.jsonl
```

The ten reported categories are OP, CR, CS, ATP, EU, TR, PR, SU, ACP, and CT.
The builder includes all question IDs by default. For comparison with the
2,498-question evaluation scope, explicitly exclude the two unavailable-media
questions using repeated `--exclude_id` arguments:

```text
--exclude_id Real-Time Visual Understanding_sample_332_4
--exclude_id Real-Time Visual Understanding_sample_332_5
```

Quote each ID when passing it in a shell. Disclose any exclusions with results;
the runner does not silently skip questions.

## Output and scoring

Manifests are JSON lists containing `id`, `video_path`, `question`, and
`timestamp`, with optional task / label metadata. Only the video, question,
and timestamp are passed to inference. The runner sorts questions by video
and timestamp, resets memory between videos, and saves raw answers and
telemetry as JSONL. Existing output files are not overwritten.

The scorer checks complete, unique manifest coverage and reports task-wise
and macro-average accuracies under two parsing policies:

- **Historical benchmark parsing:** OVO checks the answer-label occurrence
  in the response; StreamingBench applies its option-extraction rules.
- **Strict parsing:** the response must begin with a recognized option letter.

The OVO occurrence rule can overcount verbose responses. Report which policy
is used and do not mix scores from different parsing rules. Likewise, match
frame sampling, source-video coverage, token budgets, and generation settings
when comparing methods. Full-video ingestion differs from the default
query-time sampling window; see [Implementation Notes](IMPLEMENTATION.md).
