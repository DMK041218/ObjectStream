#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from objectstream.benchmark_manifests import prepare

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--suite',choices=('ovo-realtime','ovo-backward','streaming'),required=True)
    p.add_argument('--annotations',required=True)
    p.add_argument('--video_dir',required=True,help='OVO src_videos, or StreamingBench videos directory')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--exclude_id',action='append',default=[],help='Explicit exclusions only; default none')
    a=p.parse_args()
    rows=prepare(a.suite,a.annotations,a.video_dir,a.exclude_id)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(rows,f,indent=2)
    print(json.dumps({'questions':len(rows),'videos':len({r['video_path'] for r in rows}),
        'excluded_ids':a.exclude_id,'output':str(a.output)}))
