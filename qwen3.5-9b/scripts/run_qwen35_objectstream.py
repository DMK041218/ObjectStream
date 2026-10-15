#!/usr/bin/env python3
"""One query or a chronological JSON manifest; outputs immutable raw answers."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', default='Qwen/Qwen3.5-9B')
    p.add_argument('--video_path')
    p.add_argument('--question')
    p.add_argument('--timestamp', type=float)
    p.add_argument('--manifest', type=Path, help='JSON list: id,video_path,question,timestamp; optional label/task')
    p.add_argument('--output', type=Path)
    p.add_argument('--local_files_only', action='store_true')
    p.add_argument('--cards', type=int, choices=(1, 2), default=1)
    p.add_argument('--mode', choices=('objectstream', 'native'), default='objectstream')
    p.add_argument('--dry_run', action='store_true')
    a = p.parse_args()
    if a.manifest:
        rows = json.loads(a.manifest.read_text())
    else:
        if a.video_path is None or a.question is None or a.timestamp is None:
            p.error('Specify --manifest or all of --video_path,--question,--timestamp')
        rows = [{'id': 'query0', 'video_path': a.video_path, 'question': a.question, 'timestamp': a.timestamp}]
    if len({str(r['id']) for r in rows}) != len(rows):
        p.error('Duplicate query IDs')
    if any(r['timestamp'] < 0 for r in rows):
        p.error('Negative timestamp')
    rows = sorted(rows, key=lambda r:(r['video_path'], float(r['timestamp']), str(r['id'])))
    if a.dry_run:
        print(json.dumps({'dry_run':True, 'queries':len(rows),'first_query':rows[0] if rows else None,
            'model':a.model_path,'mode':a.mode,
            'recent_frames':8,'medium_frames':256,'max_entities':256,'tokens_per_entity':32,
            'max_new_tokens':128,'sampling':'causal end-anchored1fps; max256 NEW frames per update',
            'note':'No weights, media or GPU loaded.'},indent=2))
        return
    if not rows:
        p.error('Empty manifest')
    handle = None
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        handle = a.output.open('x')  # Never overwrite an earlier run.
    import torch
    if torch.cuda.device_count() != a.cards:
        raise ValueError('Set CUDA_VISIBLE_DEVICES to exactly --cards devices')
    torch.manual_seed(1234)
    torch.set_num_threads(4)
    from objectstream.qwen35_runner import Qwen35ObjectStream
    runner = Qwen35ObjectStream(a.model_path, cards=a.cards, mode=a.mode,
        local_files_only=a.local_files_only)
    previous = None
    try:
        for row in rows:
            if row['video_path'] != previous:
                runner.reset()
                previous = row['video_path']
            # Labels and task metadata are deliberately NOT forwarded.
            result = runner.answer(row['video_path'], row['question'], float(row['timestamp']))
            record = {**row, **result}
            line = json.dumps(record, ensure_ascii=False)
            if handle:
                handle.write(line+'\n');handle.flush()
            print(line, flush=True)
    finally:
        if handle:handle.close()


if __name__ == '__main__':
    main()
