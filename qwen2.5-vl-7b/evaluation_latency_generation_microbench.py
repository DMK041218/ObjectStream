#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import queue
import statistics
import threading
import time
from pathlib import Path

import torch
from moviepy.editor import VideoFileClip
from transformers import TextIteratorStreamer

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ovo_ablation_work.evaluation.ovobench import backward_tasks, forward_tasks, realtime_tasks  # noqa: E402
from qwen_vl_utils_objectstream import process_vision_info  # noqa: E402
from qwen2_5_vl_objectstream import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor  # noqa: E402

MIN_PIXELS = 16 * 28 * 28
LONG_PROMPT = (
    "Describe the video segment in detail. Include the visible scene, people or objects, "
    "important actions, temporal changes, and any cues that would help answer questions "
    "about what happened. Write a coherent paragraph."
)


def reset_peak_memory() -> None:
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            try:
                torch.cuda.reset_peak_memory_stats(i)
            except Exception:
                pass


def peak_memory_gb() -> float | None:
    if not torch.cuda.is_available():
        return None
    vals = []
    for i in range(torch.cuda.device_count()):
        try:
            vals.append(torch.cuda.max_memory_allocated(i) / (1024**3))
        except Exception:
            pass
    return max(vals) if vals else None


def selected_video_paths(args: argparse.Namespace) -> list[tuple[str, str]]:
    with open(args.task_json, "r", encoding="utf-8") as f:
        task_list = json.load(f)
    tasks = set(args.task)
    out: list[tuple[str, str]] = []
    for item in task_list:
        if item.get("task") not in tasks:
            continue
        item_id = item["id"]
        if item["task"] in backward_tasks or item["task"] in realtime_tasks:
            path = osp.join(args.video_dir, "chunked_videos", f"{item_id}.mp4")
            out.append((str(item_id), path))
        elif item["task"] in forward_tasks:
            for i in range(len(item.get("test_info", []))):
                path = osp.join(args.video_dir, "chunked_videos", f"{item_id}_{i}.mp4")
                out.append((f"{item_id}_{i}", path))
        if len(out) >= args.limit:
            break
    return out[: args.limit]


def build_inputs(video_path: str, sample_id: str, model, processor, args):
    video_start = None
    if args.time_window_size is not None:
        clip = VideoFileClip(video_path)
        dur = clip.duration
        clip.close()
        video_start = max(0.0, dur - args.time_window_size)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "min_pixels": MIN_PIXELS,
                    "max_pixels": args.max_pixels,
                    "min_frames": args.min_frames,
                    "max_frames": args.max_frames,
                    "fps": args.fps,
                    "anchor_end": args.time_window_size is not None,
                    "sample_id": sample_id,
                    "video_start": video_start,
                    "video_end": None,
                },
                {"type": "text", "text": LONG_PROMPT},
            ],
        }
    ]
    t0 = time.perf_counter()
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    t_after_video = time.perf_counter()
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    ).to(model.device)
    t_after_processor = time.perf_counter()
    return inputs, {
        "video_decode_time_s": t_after_video - t0,
        "processor_time_s": t_after_processor - t_after_video,
        "input_tokens": int(inputs.input_ids.shape[1]),
    }


def run_one(video_path: str, sample_id: str, model, processor, args) -> dict:
    inputs, prep = build_inputs(video_path, sample_id, model, processor, args)
    streamer = TextIteratorStreamer(
        processor.tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
        timeout=900,
        clean_up_tokenization_spaces=False,
    )
    gen_kwargs = dict(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        do_sample=False,
        streamer=streamer,
    )
    if args.method == "objectstream":
        gen_kwargs.update(
            use_objectstream=True,
            short_frames=args.short_frames,
            medium_frames=args.medium_frames,
            save_path=args.memory_stats_path,
        )
    elif args.method == "entitymem":
        gen_kwargs.update(
            use_entitymem=True,
            short_frames=args.short_frames,
            medium_frames=args.medium_frames,
            entity_tokens_per_entity=args.entity_tokens_per_entity,
            entity_max_entities=args.entity_max_entities,
            entity_saliency_quantile=args.entity_saliency_quantile,
            entity_match_threshold=args.entity_match_threshold,
            entity_save_path=args.memory_stats_path,
            entity_temporal_burst_selection=args.entity_temporal_burst_selection,
            entity_temporal_burst_max_events=args.entity_temporal_burst_max_events,
            entity_temporal_burst_max_events_per_entity=args.entity_temporal_burst_max_events_per_entity,
            entity_temporal_burst_radius=args.entity_temporal_burst_radius,
            entity_temporal_burst_tokens_per_frame=args.entity_temporal_burst_tokens_per_frame,
        )
    chunks: list[str] = []
    first_token_time = None
    gen_error = None
    reset_peak_memory()

    def _generate():
        nonlocal gen_error
        try:
            with torch.no_grad():
                model.generate(**gen_kwargs)
        except Exception as exc:
            gen_error = repr(exc)
            try:
                streamer.text_queue.put(None)
            except Exception:
                pass

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_start = time.perf_counter()
    thread = threading.Thread(target=_generate)
    thread.start()
    while True:
        try:
            chunk = next(streamer)
        except StopIteration:
            break
        except queue.Empty:
            gen_error = "streamer timeout"
            break
        now = time.perf_counter()
        if first_token_time is None and chunk:
            first_token_time = now
        chunks.append(chunk)
    thread.join()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_end = time.perf_counter()
    response = "".join(chunks).strip()
    output_tokens = len(processor.tokenizer(response, add_special_tokens=False).input_ids)
    gen_time = t_end - t_start
    ttft = None if first_token_time is None else first_token_time - t_start
    decode_after_first = None if ttft is None else max(0.0, gen_time - ttft)
    denom = max(1, output_tokens - 1)
    tpot = None if decode_after_first is None else decode_after_first / denom
    throughput = None if decode_after_first is None else (max(0, output_tokens - 1) / decode_after_first if decode_after_first > 0 else None)
    return {
        "id": sample_id,
        "video": video_path,
        **prep,
        "generate_time_s": gen_time,
        "ttft_s": ttft,
        "tpot_s": tpot,
        "throughput_tokens_s": throughput,
        "output_tokens": output_tokens,
        "peak_mem_gb": peak_memory_gb(),
        "response_prefix": response[:200],
        "error": gen_error,
    }


def summarize(rows: list[dict]) -> dict:
    ok = [r for r in rows if not r.get("error")]

    def vals(key):
        return [r[key] for r in ok if isinstance(r.get(key), (int, float))]

    def mean(key):
        v = vals(key)
        return sum(v) / len(v) if v else None

    def p50(key):
        v = vals(key)
        return statistics.median(v) if v else None

    return {
        "samples": len(rows),
        "successful": len(ok),
        "errors": len(rows) - len(ok),
        "mean_input_tokens": mean("input_tokens"),
        "mean_output_tokens": mean("output_tokens"),
        "mean_ttft_s": mean("ttft_s"),
        "p50_ttft_s": p50("ttft_s"),
        "mean_tpot_s": mean("tpot_s"),
        "p50_tpot_s": p50("tpot_s"),
        "mean_throughput_tokens_s": mean("throughput_tokens_s"),
        "p50_throughput_tokens_s": p50("throughput_tokens_s"),
        "mean_generate_time_s": mean("generate_time_s"),
        "peak_mem_gb": max(vals("peak_mem_gb")) if vals("peak_mem_gb") else None,
    }


def parse_args():
    p = argparse.ArgumentParser(description="Long-generation TTFT/TPOT microbenchmark")
    p.add_argument("--ckpt_path", required=True)
    p.add_argument("--task_json", default="OVO-Bench/ovo_bench_new.json")
    p.add_argument("--video_dir", default="OVO-Bench")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--method", choices=["baseline", "objectstream", "entitymem"], required=True)
    p.add_argument("--task", nargs="+", default=["STU", "OJR", "ATR", "ACR", "OCR", "FPD"])
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--max_pixels", type=int, default=401408)
    p.add_argument("--min_frames", type=int, default=4)
    p.add_argument("--max_frames", type=int, default=256)
    p.add_argument("--fps", type=int, default=1)
    p.add_argument("--time_window_size", type=float, default=256)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--min_new_tokens", type=int, default=128)
    p.add_argument("--short_frames", type=int, default=8)
    p.add_argument("--medium_frames", type=int, default=64)
    p.add_argument("--entity_tokens_per_entity", type=int, default=8)
    p.add_argument("--entity_max_entities", type=int, default=64)
    p.add_argument("--entity_saliency_quantile", type=float, default=0.75)
    p.add_argument("--entity_match_threshold", type=float, default=0.72)
    p.add_argument("--entity_temporal_burst_selection", action="store_true")
    p.add_argument("--entity_temporal_burst_max_events", type=int, default=32)
    p.add_argument("--entity_temporal_burst_max_events_per_entity", type=int, default=24)
    p.add_argument("--entity_temporal_burst_radius", type=int, default=1)
    p.add_argument("--entity_temporal_burst_tokens_per_frame", type=int, default=2)
    p.add_argument("--memory_stats_path", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.memory_stats_path is None:
        args.memory_stats_path = str(out_dir / "memory_stats.jsonl")
    print(f"Loading model {args.ckpt_path} method={args.method}", flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.ckpt_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    ).eval()
    processor = Qwen2_5_VLProcessor.from_pretrained(args.ckpt_path, min_pixels=MIN_PIXELS, max_pixels=args.max_pixels)
    processor.tokenizer.model_max_length = 32768
    rows = []
    paths = selected_video_paths(args)
    with (out_dir / "timings.jsonl").open("w", encoding="utf-8") as f:
        for idx, (sample_id, video_path) in enumerate(paths, start=1):
            print(f"[{idx}/{len(paths)}] {sample_id} {video_path}", flush=True)
            try:
                row = run_one(video_path, sample_id, model, processor, args)
            except Exception as exc:
                row = {"id": sample_id, "video": video_path, "error": repr(exc)}
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(json.dumps({k: row.get(k) for k in ["id", "input_tokens", "output_tokens", "ttft_s", "tpot_s", "throughput_tokens_s", "peak_mem_gb", "error"]}, ensure_ascii=False), flush=True)
    summary = {"method": args.method, "config": vars(args), "summary": summarize(rows)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
