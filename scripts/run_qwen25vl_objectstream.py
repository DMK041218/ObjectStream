#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from objectstream import EntityMemoryConfig, QwenObjectStreamConfig
from objectstream.qwen25vl_runner import run_qwen25vl_objectstream


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Qwen2.5-VL-7B + ObjectStream runner with entity-memory config."
    )
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--model_path", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--max_pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--short_frames", type=int, default=8)
    parser.add_argument("--medium_frames", type=int, default=64)
    parser.add_argument("--use_objectstream", action="store_true", default=True)
    parser.add_argument("--no_objectstream", dest="use_objectstream", action="store_false")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--attn_implementation", default="flash_attention_2")
    parser.add_argument("--dry_run", action="store_true")

    parser.add_argument("--max_entities", type=int, default=64)
    parser.add_argument("--entity_match_threshold", type=float, default=0.72)
    parser.add_argument("--ema_decay", type=float, default=0.9)
    parser.add_argument("--saliency_quantile", type=float, default=0.75)
    parser.add_argument("--tokens_per_entity", type=int, default=8)
    parser.add_argument("--query_topk_entities", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    qwen_config = QwenObjectStreamConfig(
        model_path=args.model_path,
        fps=args.fps,
        max_pixels=args.max_pixels,
        max_new_tokens=args.max_new_tokens,
        short_frames=args.short_frames,
        medium_frames=args.medium_frames,
        use_objectstream=args.use_objectstream,
        local_files_only=args.local_files_only,
        attn_implementation=args.attn_implementation,
    )
    entity_config = EntityMemoryConfig(
        max_entities=args.max_entities,
        entity_match_threshold=args.entity_match_threshold,
        ema_decay=args.ema_decay,
        saliency_quantile=args.saliency_quantile,
        tokens_per_entity=args.tokens_per_entity,
        query_topk_entities=args.query_topk_entities,
    )
    result = run_qwen25vl_objectstream(
        video_path=args.video_path,
        question=args.question,
        qwen_config=qwen_config,
        entity_config=entity_config,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

