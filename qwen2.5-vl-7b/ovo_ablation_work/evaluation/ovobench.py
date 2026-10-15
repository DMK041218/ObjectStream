import torch
import json
import os
import os.path as osp
import math
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
import subprocess

import argparse
import logging
import cv2
import numpy as np
import pandas as pd
from moviepy.editor import VideoFileClip
from tqdm import tqdm
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from qwen_vl_utils_objectstream import process_vision_info
from qwen2_5_vl_objectstream import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor

SCORE_SCRIPT = Path(__file__).resolve().parents[2] / "evaluation/ovobench/score.py"

# Parameters
RUN_NAME = "OVO-Bench"
CKPT_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"
TASK_NAME = ["EPM", "ASI", "HLD", "STU", "OJR", "ATR", "ACR", "OCR", "FPD", "REC", "SSR", "CRR"]
TASK_JSON = "/data/junke_dont_remove/03_Evaluations/OVO-Bench/data/ovo_bench_new.json"
VIDEO_DIR = "/data/junke_dont_remove/03_Evaluations/OVO-Bench/data"
RESULT_DIR = "eval_results/ovobench"
LOG_PATH = "log/{run_name}_{curr_time}_{task}.log"
OUTPUT_JSONL = "outputs/{run_name}_{curr_time}_{task}.jsonl"
MIN_PIXELS = 16*28*28
MAX_PIXELS = 256*28*28
MIN_FRAMES = 4
MAX_FRAMES = 256
FPS = 1
# NFRAMES = 64

backward_tasks = ["EPM", "ASI", "HLD"]
realtime_tasks = ["STU", "OJR", "ATR", "ACR", "OCR", "FPD"]
forward_tasks = ["REC", "SSR", "CRR"]

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
fmt_str = "%(asctime)s %(levelname)7s | %(message)s"
fmt = logging.Formatter(fmt_str)

# helper functions
def build_prompt(task, question, options, _anno_, index):
    if task in ["EPM", "ASI", "HLD", "STU", "OJR", "ATR", "ACR", "OCR", "FPD"]:
        formatted_options = '; '.join(f'{chr(65 + i)}. {option}' for i, option in enumerate(options)) + ';'
        prompt = f"""
            Question: {question}
            Options:
            {formatted_options}
            Respond only with the letter corresponding to your chosen option (e.g., A, B, C). 
            Do not include any additional text or explanation in your response.
        """
    elif task == "REC":
        activity = _anno_["activity"]
        question = "How many times did they " + activity + "?"
        prompt = f""" 
            You're watching a video in which people may perform a certain type of action repetively. 
            The person performing this kind of action are referred to as 'they' in the following statement.
            You're task is to count how many times have different people in the video perform this kind of action in total.
            One complete motion counts as one. 
            Now, answer the following question: {question}
            Provide your answer as a single number (e.g., 0, 1, 2, 3…) indicating the total count.
            Do not include any additional text or explanation in your response.
        """
    elif task == "SSR":
        step = _anno_["test_info"][index]["step"]
        prompt = f"""
            You're watching a tutorial video which contain a sequential of steps. 
            The following is one step from the whole procedures: 
            {step}
            Your task is to determine if the man or woman in the video is currently performing this step.
            Answer only with "Yes" or "No".
            Do not include any additional text or explanation in your response.
        """

    elif task == "CRR":
        question = _anno_["question"]
        answer = _anno_["answer"]
        prompt = f"""
            You're responsible of answering questions based on the video content. 
            The following question are relevant to the latest frames, i.e. the end of the video.
            {question}
            Decide whether existing visual content, especially latest frames, i.e. frames that near the end of the video, provide enough information for answering the question.
            Answer only with "Yes" or "No".
            Do not include any additional text or explanation in your response.
        """
    return prompt


def load_burst_caption_cache(cache_path):
    cache = {}
    if not cache_path or not osp.exists(cache_path):
        return cache
    try:
        with open(cache_path) as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                key = item.get("key")
                captions = item.get("captions")
                if key and isinstance(captions, list):
                    cache[key] = captions
    except Exception as exc:
        logger.warning(f"Could not load burst caption cache {cache_path}: {exc}")
    return cache


def cache_burst_captions(cache_path, key, captions, meta):
    if not cache_path:
        return
    try:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        record = {"key": key, "captions": captions, **meta}
        with open(cache_path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")
    except Exception as exc:
        logger.warning(f"Could not append burst caption cache {cache_path}: {exc}")


def burst_caption_cache_key(video_path, video_start, video_end, max_caps, prompt_name):
    return "|".join([
        osp.abspath(video_path),
        "none" if video_start is None else f"{float(video_start):.3f}",
        "none" if video_end is None else f"{float(video_end):.3f}",
        str(int(max_caps)),
        str(prompt_name),
    ])


def trim_caption_words(text, max_words=12):
    text = re.sub(r"^[-*\d.\s]+", "", text.strip())
    text = re.sub(r"^\[?t\s*=\s*[^\]]+\]?\s*[:\-]?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip(" .")
    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words])
    return text.strip()


def select_burst_caption_times(video_path, video_start, video_end, max_caps, fps=1.0, min_gap=1.0):
    max_caps = max(0, int(max_caps))
    if max_caps <= 0:
        return []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning(f"Could not open video for burst captions: {video_path}")
        return []
    try:
        native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        duration = frame_count / native_fps if native_fps > 0 and frame_count > 0 else 0.0
        start_time = float(video_start) if video_start is not None else 0.0
        end_time = float(video_end) if video_end is not None else duration
        if end_time <= 0:
            end_time = duration
        end_time = min(end_time, duration) if duration > 0 else end_time
        if end_time <= start_time:
            return []

        sample_fps = max(0.5, min(2.0, float(fps or 1.0)))
        step = 1.0 / sample_fps
        times = np.arange(start_time, end_time + 1e-6, step, dtype=np.float32)
        if len(times) <= max_caps:
            return [round(float(t), 2) for t in times]

        prev = None
        scores = []
        for t in times:
            cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            frame = cv2.resize(frame, (96, 96), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            if prev is not None:
                scores.append((float(np.mean(np.abs(gray - prev))), float(t)))
            prev = gray

        if not scores:
            idx = np.linspace(0, len(times) - 1, num=min(max_caps, len(times)), dtype=int)
            return [round(float(times[i]), 2) for i in idx]

        selected = []
        min_gap = max(1.0, float(min_gap))
        for _, t in sorted(scores, reverse=True):
            if all(abs(t - prev_t) >= min_gap for prev_t in selected):
                selected.append(t)
            if len(selected) >= max_caps:
                break
        selected.sort()
        return [round(float(t), 2) for t in selected]
    finally:
        cap.release()


def parse_burst_caption_response(response, times, max_caps):
    captions = []
    for raw in response.splitlines():
        line = raw.strip()
        if not line:
            continue
        caption = trim_caption_words(line, max_words=12)
        if not caption or caption.lower() == "no clear visual change":
            continue
        captions.append(caption)
        if len(captions) >= max_caps:
            break
    output = []
    for idx, caption in enumerate(captions[:max_caps]):
        t = times[min(idx, len(times) - 1)] if times else None
        output.append({"time": t, "caption": caption})
    return output


def generate_burst_captions(video_path, model, processor, video_start, video_end, args):
    times = select_burst_caption_times(
        video_path,
        video_start,
        video_end,
        args.entity_temporal_burst_caption_max,
        fps=args.fps,
        min_gap=args.entity_temporal_burst_caption_min_gap,
    )
    if not times:
        return []
    time_list = ", ".join(f"{t:.1f}s" for t in times)
    caption_prompt = (
        "You are given the video history up to the current timestamp.\n"
        "Write concise factual captions for the listed visual-change times.\n\n"
        f"Times: {time_list}\n\n"
        "Rules:\n"
        "- Output one line per time, in the same order.\n"
        "- Each line must use at most 12 English words.\n"
        "- Mention the main actor/object and action or state change.\n"
        "- Do not guess identity, intent, emotion, or unseen causes.\n"
        "- Do not answer any downstream question.\n"
        "- If no clear change is visible, write: No clear visual change.\n"
    )
    messages = [{
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": video_path,
                "min_pixels": MIN_PIXELS,
                "max_pixels": MAX_PIXELS,
                "min_frames": MIN_FRAMES,
                "max_frames": min(MAX_FRAMES, int(args.entity_temporal_burst_caption_frames)),
                "fps": FPS,
                "anchor_end": video_start is not None,
                "video_start": video_start,
                "video_end": video_end,
            },
            {"type": "text", "text": caption_prompt},
        ],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    ).to(torch.device('cuda'))
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            temperature=0.0,
            use_objectstream=False,
            use_entitymem=False,
        )
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    response = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return parse_burst_caption_response(response, times, int(args.entity_temporal_burst_caption_max))


def augment_prompt_with_burst_captions(prompt, video_path, model, processor, video_start, video_end, args, caption_cache):
    if not getattr(args, "entity_temporal_burst_caption", False):
        return prompt
    key = burst_caption_cache_key(
        video_path,
        video_start,
        video_end,
        args.entity_temporal_burst_caption_max,
        args.entity_temporal_burst_caption_prompt,
    )
    captions = caption_cache.get(key)
    if captions is None:
        captions = generate_burst_captions(video_path, model, processor, video_start, video_end, args)
        caption_cache[key] = captions
        cache_burst_captions(
            args.entity_temporal_burst_caption_cache,
            key,
            captions,
            {"video_path": video_path, "video_start": video_start, "video_end": video_end},
        )
    if not captions:
        return prompt
    lines = []
    for item in captions[: int(args.entity_temporal_burst_caption_max)]:
        cap = item.get("caption", "").strip()
        if not cap:
            continue
        t = item.get("time")
        if t is None:
            lines.append(f"- {cap}")
        else:
            lines.append(f"- [t={float(t):.1f}s] {cap}")
    if not lines:
        return prompt
    prefix = "Auxiliary visual-change captions, sorted by time:\n" + "\n".join(lines)
    prefix += "\n\nUse the video as primary evidence. These captions may be incomplete.\n\n"
    return prefix + prompt


def load_task_data(task_json_path, selected_tasks):
    with open(task_json_path, 'r') as f:
        task_list = json.load(f)
    filtered_tasks = [item for item in task_list if item['task'] in selected_tasks]
    return filtered_tasks


def distribute_samples(samples, num_gpus):
    """Distribute samples across GPUs by sample count."""
    items = []
    for item in samples:
        task = item['task']
        item_id = item['id']
        if task in ["REC", "SSR", "CRR"]:
            sample_count = len(item.get('test_info', []))
        else:
            sample_count = 1
        items.append({
            'id': item_id,
            'task': task,
            'sample_count': sample_count,
            'is_forward': task in ["REC", "SSR", "CRR"],
        })

    items.sort(key=lambda x: x['sample_count'], reverse=True)

    gpu_assignments = [[] for _ in range(num_gpus)]
    gpu_sample_counts = [0] * num_gpus

    for sample in items:
        min_idx = gpu_sample_counts.index(min(gpu_sample_counts))
        gpu_assignments[min_idx].append(sample)
        gpu_sample_counts[min_idx] += sample['sample_count']

    print("\nSample assignment:")
    total_samples = sum(gpu_sample_counts)
    for i, (samples_on_gpu, count) in enumerate(zip(gpu_assignments, gpu_sample_counts)):
        task_counts = defaultdict(int)
        id_counts = defaultdict(int)
        for s in samples_on_gpu:
            task_counts[s['task']] += 1
            id_counts[s['task']] += s['sample_count']
        task_str = ', '.join(f"{task}:{count}({id_counts[task]} samples)" for task, count in sorted(task_counts.items()))
        print(f"GPU {i}: {len(samples_on_gpu)} IDs, {count} total samples - {task_str}")
    if gpu_sample_counts and max(gpu_sample_counts) > 0:
        min_count = min(gpu_sample_counts) if min(gpu_sample_counts) > 0 else 1
        balance_ratio = max(gpu_sample_counts) / min_count
        print(f"\nBalance ratio: {balance_ratio:.2f} (closer to 1 is better)")
        print(f"Avg samples per GPU: {total_samples/num_gpus:.1f}")
    return gpu_assignments


def _list_jsonl_outputs(result_dir: Path, run_name: str):
    """List valid evaluation JSONL outputs for a given run."""
    output_dir = Path(result_dir) / "outputs"
    if not output_dir.exists():
        return []
    return [
        p for p in output_dir.glob("*.jsonl")
        if p.is_file()
        and p.name.startswith(run_name)
        and "token_drop_stats" not in p.name
        and "token_memory_stats" not in p.name
    ]


def get_response(
    prompt,
    video_path,
    model,
    processor,
    video_start=None,
    video_end=None,
    anchor_end=False,
    sample_id=None,
    use_objectstream: bool = False,
    use_entitymem: bool = False,
    short_frames: int = 8,
    medium_frames: int = 16,
    entity_tokens_per_entity: int = 8,
    entity_max_entities: int = 64,
    entity_saliency_quantile: float = 0.75,
    entity_match_threshold: float = 0.72,
    entity_temporal_selection: bool = False,
    entity_temporal_boundary_selection: bool = False,
    entity_temporal_even_selection: bool = False,
    entity_temporal_novelty_selection: bool = False,
    entity_temporal_gap_reset: bool = False,
    entity_temporal_gap_reset_frames: int | None = None,
    entity_temporal_anchor_selection: bool = False,
    entity_temporal_anchor_bins: int = 8,
    entity_temporal_anchor_tokens_per_bin: int = 4,
    entity_temporal_keyframe_selection: bool = False,
    entity_temporal_keyframe_max_frames: int = 16,
    entity_temporal_keyframe_tokens_per_frame: int = 2,
    entity_temporal_keyframe_grid_selection: bool = False,
    entity_temporal_keyframe_grid_size: int = 4,
    entity_temporal_event_selection: bool = False,
    entity_temporal_event_max_frames: int = 64,
    entity_temporal_event_tokens_per_frame: int = 8,
    entity_temporal_burst_selection: bool = False,
    entity_temporal_burst_max_events: int = 256,
    entity_temporal_burst_max_events_per_entity: int = 0,
    entity_temporal_burst_radius: int = 1,
    entity_temporal_burst_tokens_per_frame: int = 16,
    entity_chunk_selection: bool = False,
    entity_chunk_size_frames: int = 64,
    entity_chunk_max_entities: int = 0,
    entity_temporal_content_encoding: bool = False,
    entity_temporal_content_scale: float = 0.02,
    entity_temporal_order_content_encoding: bool = False,
    entity_temporal_order_content_scale: float = 0.02,
    save_path: str | None = None,
    video_chunk_t: int | None = None,
):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "min_pixels": MIN_PIXELS,
                    "max_pixels": MAX_PIXELS,
                    "min_frames": MIN_FRAMES,
                    "max_frames": MAX_FRAMES,
                    "fps": FPS,
                    "anchor_end": anchor_end,
                    "sample_id": sample_id,
                    "video_start": video_start,
                    "video_end": video_end,
                    # "nframes": NFRAMES,
                },
                {
                    "type": "text",
                    "text": prompt,
                }
            ]
        }
    ]

    if hasattr(model, "generate_response"):
        return model.generate_response(messages, argparse.Namespace(
            use_entitymem=use_entitymem,
            use_objectstream=use_objectstream,
            short_frames=short_frames,
            medium_frames=medium_frames,
            entity_tokens_per_entity=entity_tokens_per_entity,
            entity_max_entities=entity_max_entities,
            entity_saliency_quantile=entity_saliency_quantile,
            entity_match_threshold=entity_match_threshold,
            entity_temporal_selection=entity_temporal_selection,
            entity_temporal_boundary_selection=entity_temporal_boundary_selection,
            entity_temporal_even_selection=entity_temporal_even_selection,
            entity_temporal_novelty_selection=entity_temporal_novelty_selection,
            entity_temporal_gap_reset=entity_temporal_gap_reset,
            entity_temporal_gap_reset_frames=entity_temporal_gap_reset_frames,
            entity_temporal_anchor_selection=entity_temporal_anchor_selection,
            entity_temporal_anchor_bins=entity_temporal_anchor_bins,
            entity_temporal_anchor_tokens_per_bin=entity_temporal_anchor_tokens_per_bin,
            entity_temporal_keyframe_selection=entity_temporal_keyframe_selection,
            entity_temporal_keyframe_max_frames=entity_temporal_keyframe_max_frames,
            entity_temporal_keyframe_tokens_per_frame=entity_temporal_keyframe_tokens_per_frame,
            entity_temporal_keyframe_grid_selection=entity_temporal_keyframe_grid_selection,
            entity_temporal_keyframe_grid_size=entity_temporal_keyframe_grid_size,
            entity_temporal_event_selection=entity_temporal_event_selection,
            entity_temporal_event_max_frames=entity_temporal_event_max_frames,
            entity_temporal_event_tokens_per_frame=entity_temporal_event_tokens_per_frame,
            entity_temporal_burst_selection=entity_temporal_burst_selection,
            entity_temporal_burst_max_events=entity_temporal_burst_max_events,
            entity_temporal_burst_max_events_per_entity=entity_temporal_burst_max_events_per_entity,
            entity_temporal_burst_radius=entity_temporal_burst_radius,
            entity_temporal_burst_tokens_per_frame=entity_temporal_burst_tokens_per_frame,
            entity_chunk_selection=entity_chunk_selection,
            entity_chunk_size_frames=entity_chunk_size_frames,
            entity_chunk_max_entities=entity_chunk_max_entities,
            entity_temporal_content_encoding=entity_temporal_content_encoding,
            entity_temporal_content_scale=entity_temporal_content_scale,
            entity_temporal_order_content_encoding=entity_temporal_order_content_encoding,
            entity_temporal_order_content_scale=entity_temporal_order_content_scale,
            save_path=save_path,
        ), max_new_tokens=128)
    
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    )
    inputs = inputs.to(torch.device('cuda'))
    generate_kwargs = dict(max_new_tokens=128)
    if use_objectstream:
        generate_kwargs.update(use_objectstream=True, short_frames=short_frames, medium_frames=medium_frames)
        if save_path is not None:
            generate_kwargs["save_path"] = save_path
    if use_entitymem:
        generate_kwargs.update(
            use_entitymem=True,
            short_frames=short_frames,
            medium_frames=medium_frames,
            entity_tokens_per_entity=entity_tokens_per_entity,
            entity_max_entities=entity_max_entities,
            entity_saliency_quantile=entity_saliency_quantile,
            entity_match_threshold=entity_match_threshold,
            entity_temporal_selection=entity_temporal_selection,
            entity_temporal_boundary_selection=entity_temporal_boundary_selection,
            entity_temporal_even_selection=entity_temporal_even_selection,
            entity_temporal_novelty_selection=entity_temporal_novelty_selection,
            entity_temporal_gap_reset=entity_temporal_gap_reset,
            entity_temporal_gap_reset_frames=entity_temporal_gap_reset_frames,
            entity_temporal_anchor_selection=entity_temporal_anchor_selection,
            entity_temporal_anchor_bins=entity_temporal_anchor_bins,
            entity_temporal_anchor_tokens_per_bin=entity_temporal_anchor_tokens_per_bin,
            entity_temporal_keyframe_selection=entity_temporal_keyframe_selection,
            entity_temporal_keyframe_max_frames=entity_temporal_keyframe_max_frames,
            entity_temporal_keyframe_tokens_per_frame=entity_temporal_keyframe_tokens_per_frame,
            entity_temporal_keyframe_grid_selection=entity_temporal_keyframe_grid_selection,
            entity_temporal_keyframe_grid_size=entity_temporal_keyframe_grid_size,
            entity_temporal_event_selection=entity_temporal_event_selection,
            entity_temporal_event_max_frames=entity_temporal_event_max_frames,
            entity_temporal_event_tokens_per_frame=entity_temporal_event_tokens_per_frame,
            entity_temporal_burst_selection=entity_temporal_burst_selection,
            entity_temporal_burst_max_events=entity_temporal_burst_max_events,
            entity_temporal_burst_max_events_per_entity=entity_temporal_burst_max_events_per_entity,
            entity_temporal_burst_radius=entity_temporal_burst_radius,
            entity_temporal_burst_tokens_per_frame=entity_temporal_burst_tokens_per_frame,
            entity_chunk_selection=entity_chunk_selection,
            entity_chunk_size_frames=entity_chunk_size_frames,
            entity_chunk_max_entities=entity_chunk_max_entities,
            entity_temporal_content_encoding=entity_temporal_content_encoding,
            entity_temporal_content_scale=entity_temporal_content_scale,
            entity_temporal_order_content_encoding=entity_temporal_order_content_encoding,
            entity_temporal_order_content_scale=entity_temporal_order_content_scale,
        )
        if save_path is not None:
            generate_kwargs["entity_save_path"] = save_path
    if video_chunk_t is not None:
        generate_kwargs["video_chunk_t"] = int(video_chunk_t)
    generated_ids = model.generate(
        **inputs,
        **generate_kwargs,
    )
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    response = output_text[0]
    return response

### Main script
def run_single(args):
    global RUN_NAME, CKPT_PATH, RESULT_DIR, TASK_JSON, VIDEO_DIR
    global LOG_PATH, OUTPUT_JSONL, MIN_PIXELS, MAX_PIXELS, MIN_FRAMES, MAX_FRAMES, FPS

    RUN_NAME = args.run_name
    CKPT_PATH = args.ckpt_path
    RESULT_DIR = args.result_dir
    TASK_JSON = args.task_json
    VIDEO_DIR = args.video_dir
    MIN_PIXELS = args.min_pixels
    MAX_PIXELS = args.max_pixels
    MIN_FRAMES = args.min_frames
    MAX_FRAMES = args.max_frames
    FPS = args.fps

    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(osp.join(RESULT_DIR, 'outputs'), exist_ok=True)
    os.makedirs(osp.join(RESULT_DIR, 'log'), exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_PATH = args.log_path or osp.join(RESULT_DIR, "log", f"{RUN_NAME}_{timestamp}.log")
    OUTPUT_JSONL = args.output_jsonl or osp.join(RESULT_DIR, "outputs", f"{RUN_NAME}_{timestamp}.jsonl")

    file_handler = logging.FileHandler(LOG_PATH)
    file_handler.setFormatter(fmt)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    lib_logger = logging.getLogger("qwen_vl_utils")
    lib_logger.setLevel(logging.INFO)
    lib_logger.addHandler(file_handler)
    lib_logger.propagate = False
    logger.propagate = False

    logger.info(f"Running {RUN_NAME} on OVO-Bench")
    logger.info(f"Checkpoint path: {CKPT_PATH}")
    logger.info(f"Result dir: {RESULT_DIR}")
    logger.info(f"Task json: {TASK_JSON}")
    logger.info(f"Video dir: {VIDEO_DIR}")
    logger.info(f"Output jsonl: {OUTPUT_JSONL}")
    logger.info(f"Min pixels: {MIN_PIXELS}")
    logger.info(f"Max pixels: {MAX_PIXELS}")
    logger.info(f"Max frames: {MAX_FRAMES}")
    logger.info(f"Min frames: {MIN_FRAMES}")
    logger.info(f"Time window size: {args.time_window_size}")
    logger.info(f"FPS: {FPS}")
    logger.info(f"Video chunk T: {args.video_chunk_t}")
    logger.info(f"Burst Caption: {args.entity_temporal_burst_caption}, max={args.entity_temporal_burst_caption_max}")

    logger.info("=" * 50)
    logger.info("Memory Configuration:")
    logger.info(f"  Model Backend: {args.model_backend}")
    logger.info(f"  Enable Memory: {args.use_objectstream}")
    if args.use_objectstream:
        logger.info(f"  Short Frames: {args.short_frames}")
        logger.info(f"  Medium Frames: {args.medium_frames}")
    else:
        logger.info("  Memory mechanism is disabled")
    logger.info(f"  Statistics Path: {args.save_path}")
    logger.info("=" * 50)

    torch.manual_seed(1234)
    logger.info("Set manual seed to 1234")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        CKPT_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    processor = Qwen2_5_VLProcessor.from_pretrained(
        CKPT_PATH,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )

    original_tokenizer_max_len = processor.tokenizer.model_max_length
    processor.tokenizer.model_max_length = 32768
    logger.info(f"Expanded tokenizer model_max_length from {original_tokenizer_max_len} to {processor.tokenizer.model_max_length}")
    logger.info(f"Load model and processor from {CKPT_PATH}")
    caption_cache = load_burst_caption_cache(args.entity_temporal_burst_caption_cache)

    with open(TASK_JSON, 'r') as f:
        task_list = json.load(f)

    start_time = time.time()
    filtered_task_list = [item for item in task_list if item['task'] in args.task]

    if args.sample_ids:
        sample_ids_set = set(int(sid) for sid in args.sample_ids)
        filtered_task_list = [item for item in filtered_task_list if item['id'] in sample_ids_set]
        logger.info(f"Processing {len(filtered_task_list)} samples with IDs: {args.sample_ids}")
    else:
        logger.info(f"Processing all {len(filtered_task_list)} samples in tasks: {args.task}")

    for item in tqdm(filtered_task_list):
        if item['task'] in backward_tasks or item['task'] in realtime_tasks:
            id, video, task, question, options, realtime, gt = \
                item['id'], item['video'], item['task'], item['question'], item['options'], item['realtime'], item['gt']
            prompt = build_prompt(
                task=task,
                question=question,
                options=options,
                _anno_=None,
                index=None,
            )

            chunk_video_path = osp.join(VIDEO_DIR, "chunked_videos", f"{id}.mp4")
            if args.time_window_size is not None:
                clip = VideoFileClip(chunk_video_path)
                video_duration = clip.duration
                clip.close()
                video_start = max(0, video_duration - args.time_window_size)
                video_end = None
            else:
                video_start = None
                video_end = None
            prompt = augment_prompt_with_burst_captions(
                prompt,
                chunk_video_path,
                model,
                processor,
                video_start,
                video_end,
                args,
                caption_cache,
            )
            response = get_response(
                prompt=prompt,
                video_path=chunk_video_path,
                model=model,
                processor=processor,
                video_start=video_start,
                video_end=video_end,
                anchor_end=(args.time_window_size is not None),
                sample_id=id,
                use_objectstream=args.use_objectstream,
                use_entitymem=args.use_entitymem,
                short_frames=args.short_frames,
                medium_frames=args.medium_frames,
                entity_tokens_per_entity=args.entity_tokens_per_entity,
                entity_max_entities=args.entity_max_entities,
                entity_saliency_quantile=args.entity_saliency_quantile,
                entity_match_threshold=args.entity_match_threshold,
                entity_temporal_selection=args.entity_temporal_selection,
                entity_temporal_boundary_selection=args.entity_temporal_boundary_selection,
                entity_temporal_even_selection=args.entity_temporal_even_selection,
                entity_temporal_novelty_selection=args.entity_temporal_novelty_selection,
                entity_temporal_gap_reset=args.entity_temporal_gap_reset,
                entity_temporal_gap_reset_frames=args.entity_temporal_gap_reset_frames,
                entity_temporal_anchor_selection=args.entity_temporal_anchor_selection,
                entity_temporal_anchor_bins=args.entity_temporal_anchor_bins,
                entity_temporal_anchor_tokens_per_bin=args.entity_temporal_anchor_tokens_per_bin,
                entity_temporal_keyframe_selection=args.entity_temporal_keyframe_selection,
                entity_temporal_keyframe_max_frames=args.entity_temporal_keyframe_max_frames,
                entity_temporal_keyframe_tokens_per_frame=args.entity_temporal_keyframe_tokens_per_frame,
                entity_temporal_keyframe_grid_selection=args.entity_temporal_keyframe_grid_selection,
                entity_temporal_keyframe_grid_size=args.entity_temporal_keyframe_grid_size,
                entity_temporal_event_selection=args.entity_temporal_event_selection,
                entity_temporal_event_max_frames=args.entity_temporal_event_max_frames,
                entity_temporal_event_tokens_per_frame=args.entity_temporal_event_tokens_per_frame,
                entity_temporal_burst_selection=args.entity_temporal_burst_selection,
                entity_temporal_burst_max_events=args.entity_temporal_burst_max_events,
                entity_temporal_burst_max_events_per_entity=args.entity_temporal_burst_max_events_per_entity,
                entity_temporal_burst_radius=args.entity_temporal_burst_radius,
                entity_temporal_burst_tokens_per_frame=args.entity_temporal_burst_tokens_per_frame,
                entity_chunk_selection=args.entity_chunk_selection,
                entity_chunk_size_frames=args.entity_chunk_size_frames,
                entity_chunk_max_entities=args.entity_chunk_max_entities,
                entity_temporal_content_encoding=args.entity_temporal_content_encoding,
                entity_temporal_content_scale=args.entity_temporal_content_scale,
                entity_temporal_order_content_encoding=args.entity_temporal_order_content_encoding,
                entity_temporal_order_content_scale=args.entity_temporal_order_content_scale,
                save_path=args.save_path,
                video_chunk_t=args.video_chunk_t,
            )

            output_dict = {
                'id': id,
                'video': video,
                'task': task,
                'question': question,
                'response': response,
                'ground_truth': chr(65 + gt),
            }

        elif item['task'] in forward_tasks:
            id, video, task, test_info = \
                item['id'], item['video'], item['task'], item['test_info']
            for i in range(len(test_info)):
                prompt = build_prompt(
                    task=task,
                    question=None,
                    options=None,
                    _anno_=item,
                    index=i,
                )
                chunk_video_path = osp.join(VIDEO_DIR, "chunked_videos", f"{id}_{i}.mp4")

                if args.time_window_size is not None:
                    clip = VideoFileClip(chunk_video_path)
                    video_duration = clip.duration
                    clip.close()
                    video_start = max(0, video_duration - args.time_window_size)
                    video_end = None
                else:
                    video_start = None
                    video_end = None
                prompt = augment_prompt_with_burst_captions(
                    prompt,
                    chunk_video_path,
                    model,
                    processor,
                    video_start,
                    video_end,
                    args,
                    caption_cache,
                )
                response = get_response(
                    prompt=prompt,
                    video_path=chunk_video_path,
                    model=model,
                    processor=processor,
                    video_start=video_start,
                    video_end=video_end,
                    anchor_end=(args.time_window_size is not None),
                    sample_id=id,
                    use_objectstream=args.use_objectstream,
                    use_entitymem=args.use_entitymem,
                    short_frames=args.short_frames,
                    medium_frames=args.medium_frames,
                    entity_tokens_per_entity=args.entity_tokens_per_entity,
                    entity_max_entities=args.entity_max_entities,
                    entity_saliency_quantile=args.entity_saliency_quantile,
                    entity_match_threshold=args.entity_match_threshold,
                    entity_temporal_selection=args.entity_temporal_selection,
                    entity_temporal_boundary_selection=args.entity_temporal_boundary_selection,
                    entity_temporal_even_selection=args.entity_temporal_even_selection,
                    entity_temporal_novelty_selection=args.entity_temporal_novelty_selection,
                    entity_temporal_gap_reset=args.entity_temporal_gap_reset,
                    entity_temporal_gap_reset_frames=args.entity_temporal_gap_reset_frames,
                    entity_temporal_anchor_selection=args.entity_temporal_anchor_selection,
                    entity_temporal_anchor_bins=args.entity_temporal_anchor_bins,
                    entity_temporal_anchor_tokens_per_bin=args.entity_temporal_anchor_tokens_per_bin,
                    entity_temporal_keyframe_selection=args.entity_temporal_keyframe_selection,
                    entity_temporal_keyframe_max_frames=args.entity_temporal_keyframe_max_frames,
                    entity_temporal_keyframe_tokens_per_frame=args.entity_temporal_keyframe_tokens_per_frame,
                    entity_temporal_keyframe_grid_selection=args.entity_temporal_keyframe_grid_selection,
                    entity_temporal_keyframe_grid_size=args.entity_temporal_keyframe_grid_size,
                    entity_temporal_event_selection=args.entity_temporal_event_selection,
                    entity_temporal_event_max_frames=args.entity_temporal_event_max_frames,
                    entity_temporal_event_tokens_per_frame=args.entity_temporal_event_tokens_per_frame,
                    entity_temporal_burst_selection=args.entity_temporal_burst_selection,
                    entity_temporal_burst_max_events=args.entity_temporal_burst_max_events,
                    entity_temporal_burst_max_events_per_entity=args.entity_temporal_burst_max_events_per_entity,
                    entity_temporal_burst_radius=args.entity_temporal_burst_radius,
                    entity_temporal_burst_tokens_per_frame=args.entity_temporal_burst_tokens_per_frame,
                    entity_chunk_selection=args.entity_chunk_selection,
                    entity_chunk_size_frames=args.entity_chunk_size_frames,
                    entity_chunk_max_entities=args.entity_chunk_max_entities,
                    entity_temporal_content_encoding=args.entity_temporal_content_encoding,
                    entity_temporal_content_scale=args.entity_temporal_content_scale,
                    entity_temporal_order_content_encoding=args.entity_temporal_order_content_encoding,
                    entity_temporal_order_content_scale=args.entity_temporal_order_content_scale,
                    save_path=args.save_path,
                    video_chunk_t=args.video_chunk_t,
                )
                item['test_info'][i]['response'] = response

            output_dict = item
        else:
            continue

        with open(OUTPUT_JSONL, 'a' if osp.exists(OUTPUT_JSONL) else 'w') as f:
            f.write(json.dumps(output_dict) + '\n')

    end_time = time.time()
    cost_time = int(end_time - start_time)
    logger.info(f"Inference cost time: {cost_time // 3600}h {(cost_time % 3600) // 60}m {cost_time % 60}s")
    jsonl_files = _list_jsonl_outputs(RESULT_DIR, RUN_NAME)
    if not jsonl_files:
        logger.warning("No JSONL outputs found, skipping scoring.")
        return 1

    score_cmd = [
        sys.executable, str(SCORE_SCRIPT),
        "--result_dir", RESULT_DIR,
        "--run_name", RUN_NAME,
    ]
    score_res = subprocess.run(score_cmd, env=os.environ.copy())
    if score_res.returncode != 0:
        logger.error(f"Scoring failed with exit code {score_res.returncode}")
        return score_res.returncode
    logger.info("Scoring completed.")
    return 0


def run_multi_gpu(args):
    result_dir = Path(args.result_dir)
    (result_dir / "outputs").mkdir(parents=True, exist_ok=True)
    (result_dir / "log").mkdir(exist_ok=True)

    samples = load_task_data(args.task_json, args.task)
    gpu_ids = args.gpu_ids if args.gpu_ids else list(range(args.num_gpus))
    if len(gpu_ids) != args.num_gpus:
        raise ValueError("--gpu_ids length must match --num_gpus")
    gpu_assignments = distribute_samples(samples, args.num_gpus)

    if args.dry_run:
        print("\nDry run: not executing workers")
        for gpu_id, items in enumerate(gpu_assignments):
            if items:
                sample_ids = [s['id'] for s in items]
                msg = f"GPU {gpu_ids[gpu_id]} sample IDs: {sample_ids[:5]}..." if len(sample_ids) > 5 else f"GPU {gpu_ids[gpu_id]} sample IDs: {sample_ids}"
                print(msg)
        return 0

    def launch_worker(worker_idx, gpu_samples):
        if not gpu_samples:
            return True
        gpu_id = gpu_ids[worker_idx]
        sample_ids = sorted({str(s['id']) for s in gpu_samples})
        gpu_tasks = sorted({s['task'] for s in gpu_samples})

        log_path = result_dir / "log" / f"{args.run_name}_gpu{worker_idx}.log"
        output_jsonl = result_dir / "outputs" / f"{args.run_name}_gpu{worker_idx}.jsonl"

        cmd = [
            sys.executable, str(Path(__file__).resolve()),
            "--model_backend", args.model_backend,
            "--run_name", args.run_name,
            "--task", *gpu_tasks,
            "--sample_ids", *sample_ids,
            "--ckpt_path", args.ckpt_path,
            "--task_json", args.task_json,
            "--video_dir", args.video_dir,
            "--min_pixels", str(args.min_pixels),
            "--max_pixels", str(args.max_pixels),
            "--min_frames", str(args.min_frames),
            "--max_frames", str(args.max_frames),
            "--fps", str(args.fps),
            "--result_dir", args.result_dir,
            "--log_path", str(log_path),
            "--output_jsonl", str(output_jsonl),
        ]

        if args.video_chunk_t is not None:
            cmd += ["--video_chunk_t", str(args.video_chunk_t)]
        if args.use_objectstream:
            cmd.append("--use_objectstream")
        if args.use_entitymem:
            cmd.append("--use_entitymem")
        if args.use_objectstream or args.use_entitymem:
            cmd += ["--short_frames", str(args.short_frames)]
            cmd += ["--medium_frames", str(args.medium_frames)]
            cmd += ["--entity_tokens_per_entity", str(args.entity_tokens_per_entity)]
            cmd += ["--entity_max_entities", str(args.entity_max_entities)]
            cmd += ["--entity_saliency_quantile", str(args.entity_saliency_quantile)]
            cmd += ["--entity_match_threshold", str(args.entity_match_threshold)]
            if args.entity_temporal_selection:
                cmd.append("--entity_temporal_selection")
            if args.entity_temporal_boundary_selection:
                cmd.append("--entity_temporal_boundary_selection")
            if args.entity_temporal_even_selection:
                cmd.append("--entity_temporal_even_selection")
            if args.entity_temporal_novelty_selection:
                cmd.append("--entity_temporal_novelty_selection")
            if args.entity_temporal_gap_reset:
                cmd.append("--entity_temporal_gap_reset")
            if args.entity_temporal_gap_reset_frames is not None:
                cmd += ["--entity_temporal_gap_reset_frames", str(args.entity_temporal_gap_reset_frames)]
            if args.entity_temporal_anchor_selection:
                cmd.append("--entity_temporal_anchor_selection")
            cmd += ["--entity_temporal_anchor_bins", str(args.entity_temporal_anchor_bins)]
            cmd += ["--entity_temporal_anchor_tokens_per_bin", str(args.entity_temporal_anchor_tokens_per_bin)]
            if args.entity_temporal_keyframe_selection:
                cmd.append("--entity_temporal_keyframe_selection")
            cmd += ["--entity_temporal_keyframe_max_frames", str(args.entity_temporal_keyframe_max_frames)]
            cmd += ["--entity_temporal_keyframe_tokens_per_frame", str(args.entity_temporal_keyframe_tokens_per_frame)]
            if args.entity_temporal_keyframe_grid_selection:
                cmd.append("--entity_temporal_keyframe_grid_selection")
            cmd += ["--entity_temporal_keyframe_grid_size", str(args.entity_temporal_keyframe_grid_size)]
            if args.entity_temporal_event_selection:
                cmd.append("--entity_temporal_event_selection")
            cmd += ["--entity_temporal_event_max_frames", str(args.entity_temporal_event_max_frames)]
            cmd += ["--entity_temporal_event_tokens_per_frame", str(args.entity_temporal_event_tokens_per_frame)]
            if args.entity_temporal_burst_selection:
                cmd.append("--entity_temporal_burst_selection")
            cmd += ["--entity_temporal_burst_max_events", str(args.entity_temporal_burst_max_events)]
            cmd += ["--entity_temporal_burst_max_events_per_entity", str(args.entity_temporal_burst_max_events_per_entity)]
            cmd += ["--entity_temporal_burst_radius", str(args.entity_temporal_burst_radius)]
            cmd += ["--entity_temporal_burst_tokens_per_frame", str(args.entity_temporal_burst_tokens_per_frame)]
            if args.entity_temporal_burst_caption:
                cmd.append("--entity_temporal_burst_caption")
            cmd += ["--entity_temporal_burst_caption_max", str(args.entity_temporal_burst_caption_max)]
            cmd += ["--entity_temporal_burst_caption_prompt", str(args.entity_temporal_burst_caption_prompt)]
            if args.entity_temporal_burst_caption_cache:
                cache_path = Path(args.entity_temporal_burst_caption_cache)
                worker_cache = cache_path.with_name(f"{cache_path.stem}.gpu{worker_idx}{cache_path.suffix or '.jsonl'}")
                cmd += ["--entity_temporal_burst_caption_cache", str(worker_cache)]
            cmd += ["--entity_temporal_burst_caption_frames", str(args.entity_temporal_burst_caption_frames)]
            cmd += ["--entity_temporal_burst_caption_min_gap", str(args.entity_temporal_burst_caption_min_gap)]
            if args.entity_chunk_selection:
                cmd.append("--entity_chunk_selection")
            cmd += ["--entity_chunk_size_frames", str(args.entity_chunk_size_frames)]
            cmd += ["--entity_chunk_max_entities", str(args.entity_chunk_max_entities)]
            if args.entity_temporal_content_encoding:
                cmd.append("--entity_temporal_content_encoding")
            cmd += ["--entity_temporal_content_scale", str(args.entity_temporal_content_scale)]
            if args.entity_temporal_order_content_encoding:
                cmd.append("--entity_temporal_order_content_encoding")
            cmd += ["--entity_temporal_order_content_scale", str(args.entity_temporal_order_content_scale)]
            if args.save_path:
                cmd += ["--save_path", args.save_path]
        if args.time_window_size is not None:
            cmd += ["--time_window_size", str(args.time_window_size)]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - GPU {gpu_id}: {len(sample_ids)} IDs, tasks {gpu_tasks}")
        result = subprocess.run(cmd, env=env)
        return result.returncode == 0

    print("Starting parallel evaluation...")
    with ThreadPoolExecutor(max_workers=args.num_gpus) as executor:
        futures = [executor.submit(launch_worker, worker_idx, gpu_samples)
                   for worker_idx, gpu_samples in enumerate(gpu_assignments)]
        results = [f.result() for f in futures]

    all_ok = bool(results) and all(results)
    jsonl_files = _list_jsonl_outputs(result_dir, args.run_name)
    if not jsonl_files:
        print("No output JSONL found, skip scoring")
        return 1

    if not all_ok:
        print("Some GPU workers failed; attempting to score completed results")
    else:
        print("========== Evaluation finished ==========")
        print(f"========== Scoring {args.run_name} ==========")

    score_cmd = [
        sys.executable, str(SCORE_SCRIPT),
        "--result_dir", args.result_dir,
        "--run_name", args.run_name,
    ]
    score_res = subprocess.run(score_cmd, env=os.environ.copy())
    if score_res.returncode != 0:
        print(f"Scoring failed with exit code {score_res.returncode}")
        return score_res.returncode
    print("========== Scoring complete ==========")
    return 0


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_backend",
        choices=["qwen25vl_objectstream"],
        default="qwen25vl_objectstream",
        help="Model backend to use for online evaluation",
    )
    parser.add_argument("--run_name", type=str, default=RUN_NAME)
    parser.add_argument("--ckpt_path", type=str, default=CKPT_PATH)
    parser.add_argument("--result_dir", type=str, default=None)
    parser.add_argument("--task_json", type=str, default=TASK_JSON)
    parser.add_argument("--video_dir", type=str, default=VIDEO_DIR)
    parser.add_argument("--min_pixels", type=int, default=MIN_PIXELS)
    parser.add_argument("--max_pixels", type=int, default=MAX_PIXELS)
    parser.add_argument("--min_frames", type=int, default=MIN_FRAMES)
    parser.add_argument("--max_frames", type=int, default=MAX_FRAMES)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--task", type=str, nargs="+", choices=TASK_NAME, default=TASK_NAME)
    parser.add_argument("--log_path", type=str, default=None)
    parser.add_argument("--output_jsonl", type=str, default=None)
    parser.add_argument("--sample_ids", type=str, nargs="+", default=None,
                        help="Specific sample IDs to process. If not provided, process all samples in the specified tasks.")
    parser.add_argument("--time_window_size", type=float, default=None,
                        help="Time window size in seconds from video end. None means use full video.")
    parser.add_argument("--video_chunk_t", type=int, default=None,
                        help="Explicit video chunk size (frames) to avoid OOM; None disables chunking.")
    parser.add_argument("--multi_gpu", action="store_true", help="Enable multi-GPU balanced mode.")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs for multi-GPU mode.")
    parser.add_argument("--gpu_ids", nargs="+", type=int, default=None, help="Physical GPU ids for multi-GPU workers.")
    parser.add_argument("--dry_run", action="store_true", help="Show task split without running workers (multi_gpu mode).")
    parser.add_argument("--use_objectstream", action="store_true", default=False,
                        help="Enable memory mechanism for video processing")
    parser.add_argument("--use_entitymem", action="store_true", default=False,
                        help="Enable entity-centric memory mechanism for video processing")
    parser.add_argument("--entity_tokens_per_entity", type=int, default=8)
    parser.add_argument("--entity_max_entities", type=int, default=64)
    parser.add_argument("--entity_saliency_quantile", type=float, default=0.75)
    parser.add_argument("--entity_match_threshold", type=float, default=0.72)
    parser.add_argument("--entity_temporal_selection", action="store_true")
    parser.add_argument("--entity_temporal_boundary_selection", action="store_true")
    parser.add_argument("--entity_temporal_even_selection", action="store_true")
    parser.add_argument("--entity_temporal_novelty_selection", action="store_true")
    parser.add_argument("--entity_temporal_gap_reset", action="store_true")
    parser.add_argument("--entity_temporal_gap_reset_frames", type=int, default=None)
    parser.add_argument("--entity_temporal_anchor_selection", action="store_true")
    parser.add_argument("--entity_temporal_anchor_bins", type=int, default=8)
    parser.add_argument("--entity_temporal_anchor_tokens_per_bin", type=int, default=4)
    parser.add_argument("--entity_temporal_keyframe_selection", action="store_true")
    parser.add_argument("--entity_temporal_keyframe_max_frames", type=int, default=16)
    parser.add_argument("--entity_temporal_keyframe_tokens_per_frame", type=int, default=2)
    parser.add_argument("--entity_temporal_keyframe_grid_selection", action="store_true")
    parser.add_argument("--entity_temporal_keyframe_grid_size", type=int, default=4)
    parser.add_argument("--entity_temporal_event_selection", action="store_true")
    parser.add_argument("--entity_temporal_event_max_frames", type=int, default=64)
    parser.add_argument("--entity_temporal_event_tokens_per_frame", type=int, default=8)
    parser.add_argument("--entity_temporal_burst_selection", action="store_true")
    parser.add_argument("--entity_temporal_burst_max_events", type=int, default=256)
    parser.add_argument("--entity_temporal_burst_max_events_per_entity", type=int, default=0)
    parser.add_argument("--entity_temporal_burst_radius", type=int, default=1)
    parser.add_argument("--entity_temporal_burst_tokens_per_frame", type=int, default=16)
    parser.add_argument("--entity_temporal_burst_caption", action="store_true",
                        help="Generate and prepend burst-window event captions")
    parser.add_argument("--entity_temporal_burst_caption_max", type=int, default=24,
                        help="Maximum burst captions per sample")
    parser.add_argument("--entity_temporal_burst_caption_prompt", type=str, default="event_short",
                        choices=["event_short"], help="Burst caption prompt preset")
    parser.add_argument("--entity_temporal_burst_caption_cache", type=str, default=None,
                        help="JSONL cache for generated burst captions")
    parser.add_argument("--entity_temporal_burst_caption_frames", type=int, default=64,
                        help="Max video frames for caption generation")
    parser.add_argument("--entity_temporal_burst_caption_min_gap", type=float, default=1.0,
                        help="Minimum seconds between selected caption burst times")
    parser.add_argument("--entity_chunk_selection", action="store_true")
    parser.add_argument("--entity_chunk_size_frames", type=int, default=64)
    parser.add_argument("--entity_chunk_max_entities", type=int, default=0)
    parser.add_argument("--entity_temporal_content_encoding", action="store_true")
    parser.add_argument("--entity_temporal_content_scale", type=float, default=0.02)
    parser.add_argument("--entity_temporal_order_content_encoding", action="store_true")
    parser.add_argument("--entity_temporal_order_content_scale", type=float, default=0.02)
    parser.add_argument("--short_frames", type=int, default=8,
                        help="Number of short-term frames (full tokens)")
    parser.add_argument("--medium_frames", type=int, default=16,
                        help="Mid-term queue upper bound")
    parser.add_argument("--save_path", type=str, default=None,
                        help="Path to save memory statistics (jsonl format)")
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    if args.result_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.result_dir = f"eval_results/ovobench/{args.run_name}_{ts}"
    if args.multi_gpu:
        exit_code = run_multi_gpu(args)
    else:
        exit_code = run_single(args)
    sys.exit(exit_code if exit_code is not None else 0)
