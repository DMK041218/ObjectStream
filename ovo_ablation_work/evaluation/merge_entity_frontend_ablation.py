#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from ovobench.score import reorder_and_rename_score_dict, score
from ovobench.ovobench import backward_tasks, forward_tasks, realtime_tasks


def mean(rows, key):
    values = [row[key] for row in rows if not row.get("error") and row.get(key) is not None]
    return statistics.mean(values) if values else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--method", required=True)
    args = parser.parse_args()
    root = Path(args.root)

    outputs, timings, memory = [], [], []
    for rank_dir in sorted(root.glob("rank*")):
        for line in (rank_dir / "outputs.jsonl").read_text().splitlines():
            outputs.append(json.loads(line))
        for line in (rank_dir / "timings.jsonl").read_text().splitlines():
            timings.append(json.loads(line))
        stats = rank_dir / "memory_stats.jsonl"
        if stats.exists():
            memory.extend(json.loads(line) for line in stats.read_text().splitlines() if line.strip())

    grouped = {"backward": [], "realtime": [], "forward": []}
    for item in outputs:
        if item.get("task") in realtime_tasks:
            grouped["realtime"].append(item)
        elif item.get("task") in backward_tasks:
            grouped["backward"].append(item)
        elif item.get("task") in forward_tasks:
            grouped["forward"].append(item)
    scores = reorder_and_rename_score_dict(score(grouped))
    successful = [row for row in timings if not row.get("error")]
    efficiency = {
        "samples": len(timings),
        "successful": len(successful),
        "errors": len(timings) - len(successful),
        "mean_ttft_ms": 1000 * mean(timings, "ttft_s") if successful else None,
        "mean_tpot_ms": 1000 * mean(timings, "tpot_s") if successful else None,
        "mean_throughput_tokens_s": mean(timings, "throughput_tokens_s"),
        "peak_mem_gb": max((row.get("peak_mem_gb") or 0) for row in successful),
        "mean_kept_visual_tokens": mean(memory, "kept_visual_tokens"),
        "mean_visual_tokens": mean(memory, "visual_tokens"),
        "mean_frontend_latency_s": mean(memory, "entity_frontend_latency_s"),
        "frontend_fallbacks": sum(bool(row.get("entity_frontend_fallback")) for row in memory),
    }
    if efficiency["mean_kept_visual_tokens"] and efficiency["mean_visual_tokens"]:
        efficiency["mean_visual_retention"] = (
            efficiency["mean_kept_visual_tokens"] / efficiency["mean_visual_tokens"]
        )
    summary = {"method": args.method, "scores": scores, "efficiency": efficiency}
    (root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
