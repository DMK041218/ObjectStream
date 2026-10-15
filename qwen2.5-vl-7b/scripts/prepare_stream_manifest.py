#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a local JSONL video QA manifest.")
    parser.add_argument("--video_dir", required=True)
    parser.add_argument("--output", default="stream_manifest.jsonl")
    parser.add_argument("--question", default="Describe this video.")
    parser.add_argument("--glob", default="*.mp4")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_dir = Path(args.video_dir).expanduser()
    rows = []
    for path in sorted(video_dir.glob(args.glob)):
        rows.append(
            {
                "video_path": str(path),
                "question": args.question,
                "answer": None,
                "source": "local",
            }
        )
    output = Path(args.output)
    with output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()

