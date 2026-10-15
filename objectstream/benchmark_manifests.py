"""Portable dataset normalization. Labels remain metadata, never inference inputs."""
import ast
import csv
import json
from pathlib import Path

OVO_REALTIME = ('STU','OJR','ATR','ACR','OCR','FPD')
OVO_BACKWARD = ('EPM','ASI','HLD')
STREAM_TASKS = {'Object Perception':'OP','Causal Reasoning':'CR','Clips Summarize':'CS',
    'Attribute Perception':'ATP','Event Understanding':'EU','Text-Rich Understanding':'TR',
    'Prospective Reasoning':'PR','Spatial Understanding':'SU','Action Perception':'ACP','Counting':'CT'}
STREAM_TEMPLATE = """You are an advanced video question-answering AI assistant. You have been provided with some frames from the video and a multiple-choice question related to the video. Your task is to carefully analyze the video and provide the best answer to question, choosing from the four options provided. Respond with only the letter (A, B, C, or D) of the correct option.

Question: {question}

Options:
{options}

The best option is:"""


def ovo_prompt(row):
    options='; '.join(f'{chr(65+i)}. {o}' for i,o in enumerate(row['options']))+';'
    return f"""
            Question: {row['question']}
            Options:
            {options}
            Respond only with the letter corresponding to your chosen option (e.g., A, B, C).\u0020
            Do not include any additional text or explanation in your response.
        """


def prepare(suite, annotations, video_dir, exclude_ids=()):
    video_dir = Path(video_dir).resolve()
    excluded = set(exclude_ids)
    if suite.startswith('ovo-'):
        tasks = OVO_REALTIME if suite == 'ovo-realtime' else OVO_BACKWARD
        rows = json.loads(Path(annotations).read_text())
        scope = [r for r in rows if r['task'] in tasks]
        normalized = [{'id':str(r['id']), 'video_path':str(video_dir/r['video']),
            'question':ovo_prompt(r), 'timestamp':float(r['realtime']),
            'label':chr(65+int(r['gt'])), 'task':r['task'], 'benchmark':suite} for r in scope]
    elif suite == 'streaming':
        rows = list(csv.DictReader(Path(annotations).open()))
        normalized = []
        for row in rows:
            uid = row['question_id']
            options=ast.literal_eval(row['options'])
            options=[o if o.startswith(f'{chr(65+i)}.') else f'{chr(65+i)}. {o}' for i,o in enumerate(options)]
            stamp=sum(float(v)*60**i for i,v in enumerate(reversed(row['time_stamp'].split(':'))))
            normalized.append({'id':uid, 'video_path':str(video_dir/f"sample_{uid.split('_')[-2]}"/'video.mp4'),
                'question':STREAM_TEMPLATE.format(question=row['question'],options='\n'.join(options)),
                'timestamp':stamp, 'label':str(row['answer']).strip().upper(),
                'task':STREAM_TASKS[row['task_type']], 'benchmark':suite})
    else:
        raise ValueError('Unknown benchmark')
    ids = [r['id'] for r in normalized]
    if len(ids) != len(set(ids)) or not excluded.issubset(set(ids)):
        raise ValueError('Duplicate IDs or requested exclusion not present in scope')
    return [r for r in normalized if r['id'] not in excluded]
