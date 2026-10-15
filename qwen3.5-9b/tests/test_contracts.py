import json
from pathlib import Path
import subprocess
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]


def test_qwen35_dry_run_is_dependency_light():
    output=subprocess.check_output([sys.executable,str(ROOT/'scripts/run_qwen35_objectstream.py'),
        '--video_path','not-a-real-video.mp4','--question','What happens?',
        '--timestamp','10','--dry_run'],text=True)
    result=json.loads(output)
    assert result['dry_run'] and result['recent_frames']==8
    assert result['tokens_per_entity']==32


def test_production_entitymem_without_transformers():
    from objectstream.backends.qwen35.entitymem.entitymem import EntityMem
    n,h=80,16
    states=torch.randn(1,n+2,h)
    ids=torch.full((1,n+2),99,dtype=torch.long);ids[0,0]=90;ids[0,-1]=91
    geometry=torch.stack([torch.arange(20).repeat_interleave(4),
        torch.tensor([0,0,1,1]).repeat(20),torch.tensor([0,1,0,1]).repeat(20)])
    positions=torch.zeros(3,1,n+2,dtype=torch.long);positions[:,0,1:-1]=geometry
    dummy=torch.zeros(1,1,n+2,h)
    memory=EntityMem(90,91,short_term_frames=8,mid_term_frames=256,max_entities=256,
                     tokens_per_entity=32,saliency_quantile=.6,temporal_burst_selection=True,
                     temporal_burst_max_events=32,temporal_burst_radius=1,temporal_burst_tokens_per_frame=2)
    result=memory.process_memory_streaming(states,positions,(dummy,dummy),ids,torch.tensor([[20,4,4]]))
    kept=result[-1][0]
    assert torch.isin(torch.arange(65,81),kept).all()  # Last4 temporal groups=8sampled frames.
    assert kept.unique().numel()==kept.numel()
    assert int(kept.min())>=0 and int(kept.max())<n+2


def test_manifest_does_not_embed_label_in_prompt(tmp_path):
    from objectstream.benchmark_manifests import prepare
    data=[{'id':1,'task':'STU','video':'x.mp4','question':'What is visible?',
           'options':['one','two','three'],'realtime':4,'gt':0}]
    source=tmp_path/'questions.json';source.write_text(json.dumps(data))
    first=prepare('ovo-realtime',source,tmp_path)
    data[0]['gt']=2;source.write_text(json.dumps(data))
    second=prepare('ovo-realtime',source,tmp_path)
    assert first[0]['question']==second[0]['question']
    assert first[0]['label']!=second[0]['label']
