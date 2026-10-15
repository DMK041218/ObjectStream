#!/usr/bin/env python3
"""CPU-only native processor/position parity; model weights are never loaded."""
import argparse
import json
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model_path',required=True)
    p.add_argument('--video_path',type=Path,required=True)
    p.add_argument('--timestamp',type=float,default=10)
    a=p.parse_args()
    import torch
    from transformers import AutoProcessor,AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model
    from objectstream.backends.qwen35.backend import Backend,pack_layout
    torch.set_num_threads(4)
    processor=AutoProcessor.from_pretrained(a.model_path,local_files_only=True)
    config=AutoConfig.from_pretrained(a.model_path,local_files_only=True)
    backend=SimpleNamespace(processor=processor)
    inputs,prefix,suffix,groups,geometry,telemetry=Backend.preprocess(backend,a.video_path,
        'Describe the video.',0,a.timestamp)
    fake_model=SimpleNamespace(config=config)
    fake_model.get_vision_position_ids=MethodType(Qwen3_5Model.get_vision_position_ids,fake_model)
    ids,pos,_,_=pack_layout(prefix,suffix,groups,torch.zeros(len(geometry),2),geometry,
        processor.video_token_id,processor.vision_end_token_id)
    native,_=Qwen3_5Model.get_rope_index(fake_model,inputs.input_ids,inputs.mm_token_type_ids,
        video_grid_thw=inputs.video_grid_thw,attention_mask=inputs.attention_mask)
    assert torch.equal(ids,inputs.input_ids)
    assert torch.equal(pos[1:],native)
    print(json.dumps({'native_input_ids_equal':True,'native_positions_equal':True,
        'visual_tokens':len(geometry),'weights_loaded':False,'GPU_used':False,
        'observed_pts':telemetry['sampling_trace']['pts']}))


if __name__=='__main__':main()
