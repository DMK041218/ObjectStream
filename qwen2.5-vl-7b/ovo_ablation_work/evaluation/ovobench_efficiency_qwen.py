#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import queue
import re
import threading
import time
from pathlib import Path

import pandas as pd
import torch
from moviepy.editor import VideoFileClip
from transformers import TextIteratorStreamer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ovo_ablation_work.evaluation.ovobench import build_prompt, backward_tasks, realtime_tasks, forward_tasks
from evaluation.ovobench.score import score, reorder_and_rename_score_dict
from qwen_vl_utils_objectstream import process_vision_info
from qwen2_5_vl_objectstream import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor

MIN_PIXELS = 16 * 28 * 28


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
    peaks = []
    for i in range(torch.cuda.device_count()):
        try:
            peaks.append(torch.cuda.max_memory_allocated(i) / (1024 ** 3))
        except Exception:
            pass
    return max(peaks) if peaks else None


def get_response_timed(prompt, video_path, model, processor, args, video_start=None, video_end=None, anchor_end=False, sample_id=None):
    messages = [{"role": "user", "content": [{"type": "video", "video": video_path, "min_pixels": MIN_PIXELS, "max_pixels": args.max_pixels, "min_frames": args.min_frames, "max_frames": args.max_frames, "fps": args.fps, "anchor_end": anchor_end, "sample_id": sample_id, "video_start": video_start, "video_end": video_end}, {"type": "text", "text": prompt}]}]
    t0 = time.perf_counter()
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    t_after_video = time.perf_counter()
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt", **video_kwargs).to(model.device)
    t_after_processor = time.perf_counter()

    streamer = TextIteratorStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=900, clean_up_tokenization_spaces=False)
    gen_kwargs = dict(**inputs, max_new_tokens=args.max_new_tokens, streamer=streamer)
    if args.method == "objectstream":
        gen_kwargs.update(use_objectstream=True, short_frames=args.short_frames, medium_frames=args.medium_frames, save_path=args.memory_stats_path)
    elif args.method in {"entitymem", "patch_topk", "dinov2_cluster", "sam_dinov2"}:
        gen_kwargs.update(
            use_entitymem=True,
            short_frames=args.short_frames,
            medium_frames=args.medium_frames,
            entity_tokens_per_entity=args.entity_tokens_per_entity,
            entity_max_entities=args.entity_max_entities,
            entity_saliency_quantile=args.entity_saliency_quantile,
            entity_match_threshold=args.entity_match_threshold,
            entity_chunk_selection=args.entity_chunk_selection,
            entity_chunk_size_frames=args.entity_chunk_size_frames,
            entity_chunk_max_entities=args.entity_chunk_max_entities,
            entity_temporal_burst_selection=True,
            entity_temporal_burst_max_events=32,
            entity_temporal_burst_radius=1,
            entity_temporal_burst_tokens_per_frame=2,
            entity_frontend=args.method if args.method != "entitymem" else "spatial_cluster",
            entity_frontend_video_path=video_path,
            entity_save_path=args.memory_stats_path,
        )
    if args.video_chunk_t is not None:
        gen_kwargs["video_chunk_t"] = args.video_chunk_t

    chunks = []
    first_token_time = None
    gen_error = None
    reset_peak_memory()
    profile = {"reset_peak_after_vision": True}
    setattr(model, "_postvision_profile", profile)

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

    t_gen_start = time.perf_counter()
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
    t_gen_end = time.perf_counter()
    try:
        delattr(model, "_postvision_profile")
    except AttributeError:
        pass

    response = "".join(chunks).strip()
    output_tokens = len(processor.tokenizer(response, add_special_tokens=False).input_ids)
    gen_time = t_gen_end - t_gen_start
    ttft = None if first_token_time is None else first_token_time - t_gen_start
    tpot = None if ttft is None else max(0.0, gen_time - ttft) / max(1, output_tokens - 1)
    throughput = None if tpot is None or tpot <= 0 else 1.0 / tpot
    postvision_start = profile.get("postvision_start_time_s")
    postvision_latency = None if postvision_start is None else t_gen_end - postvision_start
    timing = {
        "video_decode_time_s": t_after_video - t0,
        "processor_time_s": t_after_processor - t_after_video,
        "vision_time_s": profile.get("vision_time_s"),
        "generate_time_s": gen_time,
        "e2e_time_s": t_gen_end - t0,
        "postvision_latency_s": postvision_latency,
        "ttft_s": ttft,
        "tpot_s": tpot,
        "throughput_tokens_s": throughput,
        "output_tokens": output_tokens,
        "peak_mem_gb": peak_memory_gb(),
        "postvision_peak_mem_gb": peak_memory_gb(),
        "error": gen_error,
    }
    return response, timing


def iter_selected_items(task_list, tasks, sample_ids, limit):
    selected = [item for item in task_list if item["task"] in tasks]
    if sample_ids:
        ids = set(int(x) for x in sample_ids)
        selected = [item for item in selected if int(item["id"]) in ids]
    if limit is not None and limit > 0:
        selected = selected[:limit]
    return selected


def score_outputs(outputs):
    grouped = {"backward": [], "realtime": [], "forward": []}
    for item in outputs:
        task = item.get("task")
        if task in backward_tasks:
            grouped["backward"].append(item)
        elif task in realtime_tasks:
            grouped["realtime"].append(item)
        elif task in forward_tasks:
            grouped["forward"].append(item)
    return reorder_and_rename_score_dict(score(grouped))


def summarize_timings(rows):
    ok = [r for r in rows if not r.get("error")]
    def vals(k): return [r[k] for r in ok if r.get(k) is not None]
    def mean(k):
        v = vals(k)
        return sum(v) / len(v) if v else None
    def pct(k, p):
        v = sorted(vals(k))
        if not v: return None
        return v[min(len(v) - 1, max(0, int(round((len(v) - 1) * p))))]
    return {
        "samples": len(rows),
        "successful": len(ok),
        "errors": len(rows) - len(ok),
        "mean_e2e_time_s": mean("e2e_time_s"),
        "mean_generate_time_s": mean("generate_time_s"),
        "mean_postvision_latency_s": mean("postvision_latency_s"),
        "mean_vision_time_s": mean("vision_time_s"),
        "mean_ttft_s": mean("ttft_s"),
        "p50_ttft_s": pct("ttft_s", 0.5),
        "p90_ttft_s": pct("ttft_s", 0.9),
        "mean_tpot_s": mean("tpot_s"),
        "p50_tpot_s": pct("tpot_s", 0.5),
        "p90_tpot_s": pct("tpot_s", 0.9),
        "mean_output_tokens": mean("output_tokens"),
        "mean_throughput_tokens_s": mean("throughput_tokens_s"),
        "peak_mem_gb": max(vals("peak_mem_gb")) if vals("peak_mem_gb") else None,
        "postvision_peak_mem_gb": max(vals("postvision_peak_mem_gb")) if vals("postvision_peak_mem_gb") else None,
    }


def parse_args():
    p = argparse.ArgumentParser(description="OVO-Bench efficiency probe for baseline/ObjectStream/EntityMem")
    p.add_argument("--ckpt_path", required=True)
    p.add_argument("--task_json", default="OVO-Bench/ovo_bench_new.json")
    p.add_argument("--video_dir", default="OVO-Bench")
    p.add_argument("--out_dir", default="eval_results/ovobench_efficiency/entitymem")
    p.add_argument(
        "--method",
        choices=["baseline", "objectstream", "entitymem", "patch_topk", "dinov2_cluster", "sam_dinov2"],
        default="entitymem",
    )
    p.add_argument("--task", nargs="+", default=["STU", "OJR", "ATR", "ACR", "OCR", "FPD"])
    p.add_argument("--limit", type=int, default=0, help="0 runs all selected samples")
    p.add_argument("--shard_rank", type=int, default=0)
    p.add_argument("--world_size", type=int, default=1)
    p.add_argument("--sample_ids", nargs="+", default=None)
    p.add_argument("--max_pixels", type=int, default=401408)
    p.add_argument("--min_frames", type=int, default=4)
    p.add_argument("--max_frames", type=int, default=256)
    p.add_argument("--fps", type=int, default=1)
    p.add_argument("--time_window_size", type=float, default=256)
    p.add_argument("--video_chunk_t", type=int, default=None)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--short_frames", type=int, default=8)
    p.add_argument("--medium_frames", type=int, default=64)
    p.add_argument("--entity_tokens_per_entity", type=int, default=8)
    p.add_argument("--entity_max_entities", type=int, default=64)
    p.add_argument("--entity_saliency_quantile", type=float, default=0.75)
    p.add_argument("--entity_match_threshold", type=float, default=0.72)
    p.add_argument("--entity_chunk_selection", action="store_true")
    p.add_argument("--entity_chunk_size_frames", type=int, default=64)
    p.add_argument("--entity_chunk_max_entities", type=int, default=0)
    p.add_argument("--memory_stats_path", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.memory_stats_path is None:
        args.memory_stats_path = str(out_dir / "memory_stats.jsonl")
    print(f"Loading model {args.ckpt_path} method={args.method}", flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.ckpt_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map="auto").eval()
    processor = Qwen2_5_VLProcessor.from_pretrained(args.ckpt_path, min_pixels=MIN_PIXELS, max_pixels=args.max_pixels)
    processor.tokenizer.model_max_length = 32768
    with open(args.task_json) as f:
        task_list = json.load(f)
    items = iter_selected_items(task_list, args.task, args.sample_ids, args.limit)
    items = items[args.shard_rank :: args.world_size]
    outputs = []
    timing_rows = []
    output_path = out_dir / "outputs.jsonl"
    timing_path = out_dir / "timings.jsonl"
    with output_path.open("w") as fout, timing_path.open("w") as ft:
        for idx, item in enumerate(items, start=1):
            item_id = item["id"]
            item_task = item["task"]
            print(f"[{idx}/{len(items)}] id={item_id} task={item_task}", flush=True)
            try:
                if item["task"] in backward_tasks or item["task"] in realtime_tasks:
                    prompt = build_prompt(item["task"], item["question"], item["options"], None, None)
                    chunk_video_path = osp.join(args.video_dir, "chunked_videos", f"{item_id}.mp4")
                    video_start = video_end = None
                    if args.time_window_size is not None:
                        clip = VideoFileClip(chunk_video_path)
                        dur = clip.duration
                        clip.close()
                        video_start = max(0, dur - args.time_window_size)
                    response, timing = get_response_timed(prompt, chunk_video_path, model, processor, args, video_start=video_start, video_end=video_end, anchor_end=(args.time_window_size is not None), sample_id=item["id"])
                    out = {"id": item["id"], "video": item["video"], "task": item["task"], "question": item["question"], "response": response, "ground_truth": chr(65 + item["gt"])}
                else:
                    out = dict(item)
                    timings = []
                    for i in range(len(item["test_info"])):
                        prompt = build_prompt(item["task"], None, None, item, i)
                        chunk_video_path = osp.join(args.video_dir, "chunked_videos", f"{item_id}_{i}.mp4")
                        video_start = video_end = None
                        if args.time_window_size is not None:
                            clip = VideoFileClip(chunk_video_path)
                            dur = clip.duration
                            clip.close()
                            video_start = max(0, dur - args.time_window_size)
                        response, timing = get_response_timed(prompt, chunk_video_path, model, processor, args, video_start=video_start, video_end=video_end, anchor_end=(args.time_window_size is not None), sample_id=item["id"])
                        out["test_info"][i]["response"] = response
                        timings.append(timing)
                    timing = {"e2e_time_s": sum(t.get("e2e_time_s", 0) for t in timings), "generate_time_s": sum(t.get("generate_time_s", 0) for t in timings), "ttft_s": sum(t.get("ttft_s", 0) for t in timings if t.get("ttft_s") is not None), "tpot_s": None, "output_tokens": sum(t.get("output_tokens", 0) for t in timings), "peak_mem_gb": max((t.get("peak_mem_gb") or 0) for t in timings), "error": next((t.get("error") for t in timings if t.get("error")), None)}
                outputs.append(out)
                timing.update({"id": item["id"], "task": item["task"]})
            except Exception as exc:
                out = {"id": item.get("id"), "task": item.get("task"), "response": None, "error": repr(exc)}
                timing = {"id": item.get("id"), "task": item.get("task"), "error": repr(exc)}
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            fout.flush()
            ft.write(json.dumps(timing, ensure_ascii=False) + "\n")
            ft.flush()
            timing_rows.append(timing)
    scores = score_outputs(outputs)
    efficiency = summarize_timings(timing_rows)
    summary = {
        "method": args.method,
        "tasks": args.task,
        "limit": args.limit,
        "shard_rank": args.shard_rank,
        "world_size": args.world_size,
        "scores": scores,
        "efficiency": efficiency,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
