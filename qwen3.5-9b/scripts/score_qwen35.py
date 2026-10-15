#!/usr/bin/env python3
"""Report both the historical run's rule and strict leading-option accuracy."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import statistics


def legacy_stream(response):
    for pattern in [r'option\s*([A-D])',r'([A-D])\s*is\s*the\s*best',r'answer\s*is\s*([A-D])',
                    r'([A-D])\s*\)',r'^([A-D])$',r'option is\s*([A-D])',r'\(([A-D])\)']:
        m=re.search(pattern,response.strip(),re.I)
        if m:return m.group(1).upper()
    m=re.search(r'[A-D]',response.strip())
    return m.group(0) if m else ''


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--answers',type=Path,required=True)
    a=p.parse_args()
    expected=json.loads(a.manifest.read_text())
    answers=[json.loads(line) for line in a.answers.read_text().splitlines()]
    ids=[r['id'] for r in answers]
    if len(ids)!=len(set(ids)) or set(ids)!={r['id'] for r in expected}:
        raise ValueError('Incomplete/duplicate evaluation; do not publish a partial score')
    lookup={r['id']:r for r in expected}
    groups=defaultdict(list)
    for r in answers:
        row=lookup[r['id']]
        for field in ('question','video_path','timestamp','label','task','benchmark'):
            assert r[field]==row[field],(r['id'],field)
        text=r['answer'];label=row['label']
        strict=re.match(r'^\s*\(?([A-D])(?:[.)!:,]|\s*(?:\n|$))',text)
        legacy=(label in text) if row['benchmark'].startswith('ovo-') else legacy_stream(text)==label
        groups[row['task']].append((bool(legacy),bool(strict and strict.group(1)==label)))
    scores={task:{'n':len(values),'historical_accuracy':100*sum(v[0] for v in values)/len(values),
        'strict_leading_option_accuracy':100*sum(v[1] for v in values)/len(values)} for task,values in sorted(groups.items())}
    print(json.dumps({'questions':len(answers),'task_scores':scores,
        'historical_task_macro_accuracy':statistics.mean(v['historical_accuracy'] for v in scores.values()),
        'strict_leading_option_task_macro_accuracy':statistics.mean(v['strict_leading_option_accuracy'] for v in scores.values()),
        'note':'Historical OVO rule checks letter occurrence; report this caveat, never mix scoring protocols.'},indent=2))


if __name__=='__main__':main()
