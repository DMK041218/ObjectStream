"""Training-free Qwen3.5 integration with real timestamps and per-frame wrappers.

The official hybrid decoder and visual encoder are not patched. Sparse memory
uses explicit four-axis decoder positions: sequential text positions plus the
native temporal/height/width layout. CPU tests require full-retention parity.
"""
import hashlib
import json
import time
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
from transformers.video_utils import VideoMetadata

from ._sampling import capture_decord_samples
from ._video import _read_video_decord
from .entitymem.entitymem import EntityMem

def digest(x):
    return hashlib.sha256(x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def recent_visual_mask(geometry, sampled_frames, temporal_patch_size):
    """EntityMem budgets sampled frames, not encoded temporal groups.

    Each Qwen3.5 temporal group encodes two sampled frames. Match the
    existing EntityMem conversion; do not silently double the recent budget.
    """
    if sampled_frames == 0:
        return torch.zeros(len(geometry),dtype=torch.bool,device=geometry.device)
    count = max(1, int(sampled_frames) // max(1, int(temporal_patch_size)))
    times = torch.unique(geometry[:, 0], sorted=True)
    return torch.isin(geometry[:, 0], times[-count:])


def pack_layout(prefix, suffix, frames, features, geometry, video_id, end_id):
    """Keep each surviving frame's timestamp and boundaries; retain local grid.

    geometry columns: frame identity, local row, local column. Each frame has
    a full-canvas H/W, even when its surviving patches do not touch the edges.
    """
    device = features.device
    ids, positions, visual_indices = [], [], []
    offset = 0
    length = 0

    def text(tokens):
        nonlocal offset, length
        tokens = tokens.to(device=device, dtype=torch.long)
        ids.append(tokens)
        positions.append(torch.arange(offset, offset + len(tokens), device=device).expand(3, -1))
        offset += len(tokens)
        length += len(tokens)

    text(prefix)
    for i, frame in enumerate(frames):
        local = (geometry[:, 0] == i).nonzero().flatten()
        if not len(local):
            continue
        text(frame['lead'])
        ids.append(torch.full((len(local),), video_id, device=device, dtype=torch.long))
        rowcol = geometry[local, 1:].T.to(device)
        positions.append(torch.cat([torch.zeros((1, len(local)), device=device, dtype=torch.long), rowcol]) + offset)
        visual_indices.extend(range(length, length + len(local)))
        length += len(local)
        offset += max(frame['h'], frame['w'])
        text(torch.tensor([end_id]))
    text(suffix)
    input_ids = torch.cat(ids).unsqueeze(0)
    vision_positions = torch.cat(positions, dim=1).unsqueeze(1)
    text_positions = torch.arange(length, device=device).reshape(1, 1, -1)
    positions4 = torch.cat([text_positions, vision_positions])
    visual_indices = torch.tensor(visual_indices, device=device, dtype=torch.long)
    # Features are sorted into the same frame order as the reconstructed prompt.
    order = torch.cat([(geometry[:, 0] == i).nonzero().flatten() for i in range(len(frames))])
    return input_ids, positions4, visual_indices, features[order]


class Backend:
    def __init__(self, model_path, cards=1, local_files_only=True):
        self.recent_frames=8
        self.last_selection={}
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=local_files_only)
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_path, local_files_only=local_files_only, dtype=torch.bfloat16,
            attn_implementation='sdpa', device_map={'': 0} if cards == 1 else 'auto',
            **({'max_memory': {i: '12GiB' for i in range(cards)}} if cards > 1 else {}),
        ).eval()
        if self.model.config.vision_config.deepstack_visual_indexes:
            raise ValueError('This Qwen3.5 adapter requires no DeepStack; audit a separate path otherwise')
        if self.model.config.vision_config.temporal_patch_size != 2:
            raise ValueError('This adapter requires the audited two-frame temporal patch size')
        self.device = self.model.get_input_embeddings().weight.device
        self.cache = {}

    def sync(self):
        for i in range(torch.cuda.device_count()):
            torch.cuda.synchronize(i)

    def preprocess(self, video, prompt, start, end):
        # Decode using the existing causal 1-fps/end-anchored sampling, but do
        # NOT perform the old Qwen2.5 factor-28 resize/normalization.
        element = dict(video=str(video), fps=1, min_frames=4, max_frames=256,
                       anchor_end=True, video_start=start, video_end=end)
        provided = getattr(self, '_provided_segment', None)
        if provided is not None:
            # Evaluation-adapter-only override: common full-video 1fps plan,
            # bounded batches, actual source indices/PTS. Native processor,
            # positions and all compression algorithms remain unchanged.
            frames, trace = provided
            if len(trace['pts']) != len(frames) or len(trace['indices']) != len(frames):
                raise ValueError('Provided segment trace misaligned')
        else:
            with capture_decord_samples() as traces:
                if end-start < 2:
                    import decord
                    reader = decord.VideoReader(str(video))
                    index = min(len(reader)-1, max(0, int(end*reader.get_avg_fps())))
                    frames = torch.from_numpy(reader.get_batch([index]*4).asnumpy()).permute(0,3,1,2)
                else:
                    frames, _ = _read_video_decord(element)
            if len(traces) != 1:
                raise ValueError('Missing actual Decord sampling trace')
            trace = traces[0]
        pts = trace['pts']
        if any(p < 0 or p > end + 1e-4 for p in pts) or pts != sorted(pts):
            raise ValueError('Future or unordered source frames')
        metadata = VideoMetadata(total_num_frames=max(trace['indices']) + 1,
                                 fps=trace['source_fps'], frames_indices=trace['indices'])
        messages = [{'role': 'user', 'content': [
            {'type': 'video', 'video': str(video)}, {'type': 'text', 'text': prompt}]}]
        template = self.processor.apply_chat_template(messages, tokenize=False,
                             add_generation_prompt=True, enable_thinking=False)
        inputs = self.processor(text=[template], videos=[frames], video_metadata=[metadata],
                  videos_kwargs={'do_sample_frames': False,
                                 'cap_pixels_per_frame':False,
                                 # Native Qwen3VLVideoProcessor's size budget
                                 # is TOTAL clip pixels, not pixels/frame.
                                 'size': {'shortest_edge': 12544*len(frames),
                                          'longest_edge': 200704*len(frames)}},
                  padding=True, return_tensors='pt')
        ids = inputs.input_ids[0]
        marker = self.processor.vision_start_token + self.processor.video_token + self.processor.vision_end_token
        if template.count(marker) != 1:
            raise ValueError('Unexpected native video placeholder')
        # Transformers 5.18 replaces the video_pad placeholder only; the
        # chat template's outer vision boundary remains as native text.
        prefix_text = template.split(self.processor.video_token)[0]
        prefix = torch.tensor(self.processor.tokenizer.encode(prefix_text, add_special_tokens=False))
        if not torch.equal(ids[:len(prefix)], prefix):
            raise ValueError('Native prompt prefix tokenization changed at timestamp boundary')
        vs, ve, vp = self.processor.vision_start_token_id, self.processor.vision_end_token_id, self.processor.video_token_id
        starts = (ids == vs).nonzero().flatten()
        starts = starts[(starts+1 < len(ids)) & (ids[(starts+1).clamp(max=len(ids)-1)] == vp)]
        all_ends = (ids == ve).nonzero().flatten()
        ends = torch.tensor([int(all_ends[all_ends>s][0]) for s in starts])
        t, raw_h, raw_w = map(int, inputs.video_grid_thw[0])
        h, w = raw_h // 2, raw_w // 2
        if len(starts) != t or len(ends) != t or len(pts) != t * 2:
            raise ValueError(f'Native wrappers/PTS disagree: starts={len(starts)}, ends={len(ends)}, grid_t={t}, pts={len(pts)}; decoded={self.processor.decode(ids)[:400]}')
        groups = []
        previous = len(prefix)
        for i, (s, e) in enumerate(zip(starts.tolist(), ends.tolist())):
            if not torch.all(ids[s+1:e] == vp) or e-s-1 != h*w:
                raise ValueError('Unexpected native visual placeholder layout')
            groups.append(dict(lead=ids[previous:s+1].clone(), h=h, w=w,
                               time_us=round((pts[2*i]+pts[2*i+1])/2*1e6)))
            previous = e+1
        suffix = ids[previous:].clone()
        geometry = torch.stack([torch.arange(t).repeat_interleave(h*w),
                    torch.arange(h).repeat_interleave(w).repeat(t),
                    torch.arange(w).repeat(h).repeat(t)], dim=1)
        telemetry = {'sampling_trace': trace, 'query_timestamp_sec': end, 'segment_start_sec': start,
            'input_fingerprints': {'input_ids': digest(inputs.input_ids),
                                   'video_pixels': digest(inputs.pixel_values_videos),
                                   'grid': inputs.video_grid_thw.tolist()},
            'timestamp_policy': 'actual_source_frame_indices/source_fps',
            'resize_policy': 'official_processor_only_factor32', 'thinking': False}
        return inputs, prefix, suffix, groups, geometry, telemetry

    @torch.inference_mode()
    def encode(self, inputs):
        pixels = inputs.pixel_values_videos.to(self.device, torch.bfloat16)
        grid = inputs.video_grid_thw.to(self.device)
        # The same official full encoding is used by native and manual paths.
        outputs = self.model.get_video_features(pixels, grid, return_dict=True)
        return torch.cat(outputs.pooler_output).to(self.device)

    def select(self, features, geometry, groups):
        n, hidden = features.shape
        vp = self.processor.video_token_id
        vs, ve = self.processor.vision_start_token_id, self.processor.vision_end_token_id
        entity = EntityMem(vs, ve, short_term_frames=self.recent_frames, mid_term_frames=256,
                    max_entities=256, tokens_per_entity=32, saliency_quantile=.60,
                    match_threshold=.72, temporal_burst_selection=True,
                    temporal_burst_max_events=32, temporal_burst_max_events_per_entity=0,
                    temporal_burst_radius=1, temporal_burst_tokens_per_frame=2)
        entity.save_path = None
        entity._compute_grid_hw = lambda *_: (max(f['h'] for f in groups), max(f['w'] for f in groups))
        synthetic = torch.cat([torch.zeros((1, hidden), device=features.device, dtype=features.dtype),
                               features, torch.zeros((1, hidden), device=features.device, dtype=features.dtype)])[None]
        ids = torch.full((1, n+2), vp, device=features.device, dtype=torch.long)
        ids[0, 0], ids[0, -1] = vs, ve
        positions = torch.zeros((3, 1, n+2), device=features.device, dtype=torch.long)
        positions[:, 0, 1:-1] = geometry.T
        zeros = torch.zeros((1, 1, n+2, hidden), device=features.device, dtype=features.dtype)
        result = entity.process_memory_streaming(hidden_states=synthetic, position_ids=positions,
                position_embeddings=(zeros, zeros), input_ids=ids,
                video_grid_thw=torch.tensor([[len(groups), max(f['h'] for f in groups), max(f['w'] for f in groups)]],
                                           device=features.device))
        keep = result[-1][0]
        keep = keep[(keep > 0) & (keep < n+1)] - 1
        keep = keep.sort().values
        self.last_selection=entity.last_records[0]
        recent = recent_visual_mask(geometry, entity.short_term_frames, entity.temporal_patch_size)
        if not torch.isin(recent.nonzero().flatten(), keep).all():
            raise ValueError('Recent visual tokens were lost')
        return keep

    @torch.inference_mode()
    def answer(self, video, prompt, end, mode, parity=False):
        query_started = time.perf_counter()
        key = str(video)
        state = self.cache.get(key) if mode == 'objectstream' else None
        start = state['end'] if state else 0.0
        if end < start:
            raise ValueError('Queries must be strictly chronological within each video')
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(i)
        reuse = state is not None and end == start
        if reuse:
            # Multiple questions at the same observation time share exactly
            # the same generic memory; do not encode or compress it again.
            messages=[{'role':'user','content':[{'type':'video','video':str(video)},
                                               {'type':'text','text':prompt}]}]
            template=self.processor.apply_chat_template(messages,tokenize=False,
                                         add_generation_prompt=True,enable_thinking=False)
            marker=self.processor.vision_start_token+self.processor.video_token+self.processor.vision_end_token
            if template.count(marker)!=1:raise ValueError('Unexpected same-time query template')
            suffix=torch.tensor(self.processor.tokenizer.encode(template.split(self.processor.video_token)[1],add_special_tokens=False))
            prefix=state['prefix']
            inputs=None
            telemetry={'query_timestamp_sec':end,'segment_start_sec':start,'reuse_at_equal_time':True,
                       'timestamp_policy':'cached_actual_source_time_wrappers','thinking':False}
        else:
            inputs, prefix, suffix, groups, geometry, telemetry = self.preprocess(video, prompt, start, end)
        telemetry['video_decode_preprocess_s'] = time.perf_counter()-query_started
        self.model.model.rope_deltas = None
        if mode == 'native':
            kwargs = {k: v.to(self.device) if torch.is_tensor(v) else v for k,v in inputs.items()}
            kept = int((inputs.input_ids == self.processor.video_token_id).sum())
            previous = 0
            telemetry['video_encoding_s'] = None  # integrated in official generation
        else:
            self.sync(); encoding_started = time.perf_counter()
            features = state['features'].to(self.device) if reuse else self.encode(inputs)
            self.sync(); telemetry['video_encoding_s'] = time.perf_counter()-encoding_started
            geometry = state['geometry'].to(self.device) if reuse else geometry.to(self.device)
            if reuse:groups=state['groups']
            previous = 0
            if state is not None:
                previous = len(state['features'])
                if not reuse:
                    geometry[:, 0] += len(state['groups'])
                    features = torch.cat([state['features'].to(self.device), features])
                    geometry = torch.cat([state['geometry'].to(self.device), geometry])
                    groups = state['groups'] + groups
            # Explicit order after segment merge; no prompt-layout coordinates
            # are used as physical time. Equal-time groups remain distinct.
            ordering = sorted(range(len(groups)), key=lambda i: (groups[i]['time_us'], i))
            inverse = torch.empty(len(groups), dtype=torch.long, device=self.device)
            inverse[torch.tensor(ordering, device=self.device)] = torch.arange(len(groups), device=self.device)
            geometry[:, 0] = inverse[geometry[:, 0]]
            groups = [groups[i] for i in ordering]
            if max(f['time_us'] for f in groups) > round(end*1e6)+100:
                raise ValueError('Cached memory contains future evidence')
            self.sync(); memory_started = time.perf_counter()
            candidate_count = len(features)
            keep = torch.arange(len(features), device=self.device) if parity or reuse else self.select(features, geometry, groups)
            recent = recent_visual_mask(geometry, self.recent_frames, 2)
            telemetry['recent_sampled_frame_budget'] = self.recent_frames
            telemetry['recent_temporal_group_budget'] = self.recent_frames//2
            telemetry['recent_candidate_visual_tokens'] = int(recent.sum())
            telemetry['recent_kept_visual_tokens'] = int(recent[keep].sum())
            telemetry['kept_old_memory_tokens'] = int((keep < previous).sum())
            telemetry['selection_record']=self.last_selection
            features, geometry = features[keep], geometry[keep]
            ids, pos4, visual, sorted_features = pack_layout(prefix, suffix, groups, features, geometry,
                                      self.processor.video_token_id, self.processor.vision_end_token_id)
            if parity:
                official, _ = self.model.model.get_rope_index(inputs.input_ids.to(self.device),
                   inputs.mm_token_type_ids.to(self.device), video_grid_thw=inputs.video_grid_thw.to(self.device),
                   attention_mask=inputs.attention_mask.to(self.device))
                if not torch.equal(ids, inputs.input_ids.to(self.device)) or not torch.equal(pos4[1:], official):
                    raise ValueError('Full-retention prompt/position parity failed')
            embeds = self.model.get_input_embeddings()(ids)
            embeds[0, visual] = sorted_features
            kwargs = {'input_ids': ids, 'inputs_embeds': embeds,
                      'attention_mask': torch.ones_like(ids), 'position_ids': pos4}
            kept = len(features)
            # Keep only represented groups; old text wrappers accompany their
            # surviving tokens instead of accumulating empty frames forever.
            active = sorted(set(geometry[:, 0].tolist()))
            mapping = {old: new for new, old in enumerate(active)}
            cached_geometry = geometry.cpu().clone()
            cached_geometry[:, 0] = torch.tensor([mapping[x] for x in cached_geometry[:, 0].tolist()])
            if not parity:
                self.cache[key] = {'features': features.cpu(), 'geometry': cached_geometry,
                                   'groups': [groups[i] for i in active], 'prefix':prefix.cpu(), 'end': end}
            self.sync(); telemetry['memory_processing_s'] = time.perf_counter()-memory_started
            telemetry['candidate_visual_tokens'] = candidate_count
            telemetry['frame_wrapper_groups'] = len(active)
        telemetry['kept_visual_tokens'] = kept
        telemetry['previous_memory_tokens'] = previous
        if getattr(self, '_ingest_only', False):
            if mode != 'objectstream' or parity:
                raise ValueError('Ingest-only requires real compressed-memory update')
            telemetry['generation_s'] = 0.0
            telemetry['generated_tokens'] = 0
            telemetry['query_to_complete_s'] = time.perf_counter()-query_started
            telemetry['peak_allocated_gib'] = [torch.cuda.max_memory_allocated(i)/2**30 for i in range(torch.cuda.device_count())]
            telemetry['ingest_only'] = True
            return '', telemetry, None
        self.sync(); generation_started = time.perf_counter()
        output = self.model.generate(**kwargs, max_new_tokens=128, do_sample=False,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    return_dict_in_generate=True, output_scores=parity)
        self.sync()
        telemetry['generation_s'] = time.perf_counter()-generation_started
        telemetry['query_to_complete_s'] = time.perf_counter()-query_started
        prompt_length = kwargs['input_ids'].shape[-1]
        tokens = output.sequences[0, prompt_length:]
        response = self.processor.decode(tokens, skip_special_tokens=True).strip()
        telemetry['generated_tokens'] = len(tokens)
        telemetry['peak_allocated_gib'] = [torch.cuda.max_memory_allocated(i)/2**30 for i in range(torch.cuda.device_count())]
        telemetry['gpus_per_worker'] = torch.cuda.device_count()
        telemetry['latency_note'] = 'Offline query total includes decode/vision; generation_s reported separately.'
        return response, telemetry, output.scores[0].float().cpu() if parity else None
