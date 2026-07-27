"""LLaVA-OneVision inference helper with EntityMem-style video compression.

This module keeps the online benchmark scripts model-agnostic without patching
Transformers internals. It computes LLaVA-OneVision video embeddings, applies
the existing EntityMem/Burst selector on those pooled embeddings, then feeds the
compressed prompt embeddings directly to the language model.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
from transformers import LlavaOnevisionForConditionalGeneration, LlavaOnevisionProcessor

from qwen2_5_vl_objectstream.entitymem import EntityMem
from qwen_vl_utils_objectstream import process_vision_info


class LlavaOnevisionEntityMemGenerator:
    def __init__(self, ckpt_path: str, min_pixels: int, max_pixels: int, device: str = "cuda") -> None:
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            ckpt_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map="auto",
        ).eval()
        self.processor = LlavaOnevisionProcessor.from_pretrained(
            ckpt_path,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        self.tokenizer = self.processor.tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _make_entitymem(self, args: Any) -> EntityMem:
        cfg = self.model.config
        return EntityMem(
            vision_start_token_id=int(cfg.video_token_index),
            vision_end_token_id=int(cfg.image_token_index),
            short_term_frames=int(args.short_frames),
            mid_term_frames=int(args.medium_frames),
            max_entities=int(args.entity_max_entities),
            tokens_per_entity=int(args.entity_tokens_per_entity),
            saliency_quantile=float(args.entity_saliency_quantile),
            match_threshold=float(args.entity_match_threshold),
            temporal_selection=bool(getattr(args, "entity_temporal_selection", False)),
            temporal_boundary_selection=bool(getattr(args, "entity_temporal_boundary_selection", False)),
            temporal_even_selection=bool(getattr(args, "entity_temporal_even_selection", False)),
            temporal_novelty_selection=bool(getattr(args, "entity_temporal_novelty_selection", False)),
            temporal_gap_reset=bool(getattr(args, "entity_temporal_gap_reset", False)),
            temporal_gap_reset_frames=getattr(args, "entity_temporal_gap_reset_frames", None),
            temporal_anchor_selection=bool(getattr(args, "entity_temporal_anchor_selection", False)),
            temporal_anchor_bins=int(getattr(args, "entity_temporal_anchor_bins", 8)),
            temporal_anchor_tokens_per_bin=int(getattr(args, "entity_temporal_anchor_tokens_per_bin", 4)),
            temporal_keyframe_selection=bool(getattr(args, "entity_temporal_keyframe_selection", False)),
            temporal_keyframe_max_frames=int(getattr(args, "entity_temporal_keyframe_max_frames", 16)),
            temporal_keyframe_tokens_per_frame=int(getattr(args, "entity_temporal_keyframe_tokens_per_frame", 2)),
            temporal_keyframe_grid_selection=bool(getattr(args, "entity_temporal_keyframe_grid_selection", False)),
            temporal_keyframe_grid_size=int(getattr(args, "entity_temporal_keyframe_grid_size", 4)),
            temporal_event_selection=bool(getattr(args, "entity_temporal_event_selection", False)),
            temporal_event_max_frames=int(getattr(args, "entity_temporal_event_max_frames", 64)),
            temporal_event_tokens_per_frame=int(getattr(args, "entity_temporal_event_tokens_per_frame", 8)),
            temporal_burst_selection=bool(getattr(args, "entity_temporal_burst_selection", False)),
            temporal_burst_max_events=int(getattr(args, "entity_temporal_burst_max_events", 32)),
            temporal_burst_max_events_per_entity=int(getattr(args, "entity_temporal_burst_max_events_per_entity", 0)),
            temporal_burst_radius=int(getattr(args, "entity_temporal_burst_radius", 1)),
            temporal_burst_tokens_per_frame=int(getattr(args, "entity_temporal_burst_tokens_per_frame", 2)),
            chunk_selection=bool(getattr(args, "entity_chunk_selection", False)),
            chunk_size_frames=int(getattr(args, "entity_chunk_size_frames", 64)),
            chunk_max_entities=int(getattr(args, "entity_chunk_max_entities", 0)),
            temporal_content_encoding=bool(getattr(args, "entity_temporal_content_encoding", False)),
            temporal_content_scale=float(getattr(args, "entity_temporal_content_scale", 0.02)),
            temporal_order_content_encoding=bool(getattr(args, "entity_temporal_order_content_encoding", False)),
            temporal_order_content_scale=float(getattr(args, "entity_temporal_order_content_scale", 0.02)),
        )

    def _video_grid(self, num_frames: int, tokens_per_frame: int) -> tuple[int, int]:
        image_size = int(self.model.config.vision_config.image_size)
        patch_size = int(self.model.config.vision_config.patch_size)
        base_hw = image_size // patch_size
        pooled_hw = math.ceil(base_hw / 2)
        if pooled_hw * pooled_hw == tokens_per_frame:
            return pooled_hw, pooled_hw
        side = int(math.sqrt(tokens_per_frame))
        if side * side == tokens_per_frame:
            return side, side
        return 1, tokens_per_frame

    def _compress_video_features(self, video_features: torch.Tensor, args: Any) -> tuple[torch.Tensor, int, int]:
        if not getattr(args, "use_entitymem", False):
            return video_features, int(video_features.shape[1]), int(video_features.shape[1])

        batch, total_tokens, hidden = video_features.shape
        if batch != 1:
            raise ValueError("LLaVA-OneVision online helper expects batch size 1")

        num_frames = int(getattr(self, "_last_num_video_frames", 1))
        tokens_per_frame = max(1, total_tokens // max(1, num_frames))
        usable_tokens = num_frames * tokens_per_frame
        features = video_features[:, :usable_tokens, :]
        tail = video_features[:, usable_tokens:, :]
        grid_h, grid_w = self._video_grid(num_frames, tokens_per_frame)
        if grid_h * grid_w != tokens_per_frame:
            grid_h, grid_w = 1, tokens_per_frame

        cfg = self.model.config
        start_id = int(cfg.video_token_index)
        end_id = int(cfg.image_token_index)
        dummy_id = int(self.tokenizer.pad_token_id or self.tokenizer.eos_token_id or 0)

        dummy_start = torch.zeros(1, 1, hidden, dtype=features.dtype, device=features.device)
        dummy_end = torch.zeros(1, 1, hidden, dtype=features.dtype, device=features.device)
        hidden_states = torch.cat([dummy_start, features, dummy_end], dim=1)
        seq_len = hidden_states.shape[1]
        input_ids = torch.full((1, seq_len), dummy_id, dtype=torch.long, device=features.device)
        input_ids[0, 0] = start_id
        input_ids[0, -1] = end_id

        time_ids = []
        height_ids = []
        width_ids = []
        for frame_id in range(num_frames):
            for token_id in range(tokens_per_frame):
                time_ids.append(frame_id)
                height_ids.append(token_id // grid_w)
                width_ids.append(token_id % grid_w)
        pos = torch.zeros(3, 1, seq_len, dtype=torch.long, device=features.device)
        pos[0, 0, 1:-1] = torch.tensor(time_ids, dtype=torch.long, device=features.device)
        pos[1, 0, 1:-1] = torch.tensor(height_ids, dtype=torch.long, device=features.device)
        pos[2, 0, 1:-1] = torch.tensor(width_ids, dtype=torch.long, device=features.device)
        pos_emb = (
            torch.zeros(1, 1, seq_len, hidden, dtype=features.dtype, device=features.device),
            torch.zeros(1, 1, seq_len, hidden, dtype=features.dtype, device=features.device),
        )

        entitymem = self._make_entitymem(args)
        entitymem.save_path = getattr(args, "save_path", None)
        compressed_hidden, _, _, _, kept = entitymem.process_memory_streaming(
            hidden_states,
            pos,
            pos_emb,
            input_ids,
            video_grid_thw=torch.tensor([[num_frames, grid_h, grid_w]], device=features.device),
        )
        kept_visual = kept[0].to(features.device)
        kept_visual = kept_visual[(kept_visual > 0) & (kept_visual < seq_len - 1)] - 1
        kept_features = features[:, kept_visual, :]
        if tail.numel() > 0:
            kept_features = torch.cat([kept_features, tail], dim=1)
        return kept_features, int(total_tokens), int(kept_features.shape[1])

    def _prepare_prompt_embeddings(self, inputs: Any, args: Any) -> tuple[torch.Tensor, torch.Tensor, int]:
        input_ids = inputs.input_ids.to(self.model.device)
        attention_mask = inputs.attention_mask.to(self.model.device)
        embeds = self.model.get_input_embeddings()(input_ids)

        if not hasattr(inputs, "pixel_values_videos") or inputs.pixel_values_videos is None:
            return embeds, attention_mask, embeds.shape[1]

        pixel_values_videos = inputs.pixel_values_videos.to(self.model.device, dtype=torch.bfloat16)
        self._last_num_video_frames = int(pixel_values_videos.shape[1])
        video_features = self.model.get_video_features(
            pixel_values_videos,
            vision_feature_layer=self.model.config.vision_feature_layer,
            vision_feature_select_strategy=self.model.config.vision_feature_select_strategy,
        )
        video_features, original_visual, kept_visual = self._compress_video_features(video_features, args)
        image_newline = self.model.image_newline[None, None, :].repeat(video_features.shape[0], 1, 1).to(
            video_features.device,
            video_features.dtype,
        )
        video_features = torch.cat((video_features, image_newline), dim=1).to(embeds.dtype)

        video_token = int(self.model.config.video_token_index)
        pieces = []
        masks = []
        for batch_idx in range(input_ids.shape[0]):
            video_pos = (input_ids[batch_idx] == video_token).nonzero(as_tuple=True)[0]
            if video_pos.numel() == 0:
                pieces.append(embeds[batch_idx])
                masks.append(attention_mask[batch_idx])
                continue
            first = int(video_pos[0].item())
            last = int(video_pos[-1].item()) + 1
            pieces.append(torch.cat([embeds[batch_idx, :first], video_features[batch_idx], embeds[batch_idx, last:]], dim=0))
            masks.append(torch.cat([
                attention_mask[batch_idx, :first],
                torch.ones(video_features.shape[1], dtype=attention_mask.dtype, device=attention_mask.device),
                attention_mask[batch_idx, last:],
            ], dim=0))

        max_len = max(piece.shape[0] for piece in pieces)
        padded_embeds = []
        padded_masks = []
        for piece, mask in zip(pieces, masks):
            pad_len = max_len - piece.shape[0]
            if pad_len > 0:
                piece = torch.cat([piece, torch.zeros(pad_len, piece.shape[1], dtype=piece.dtype, device=piece.device)], dim=0)
                mask = torch.cat([mask, torch.zeros(pad_len, dtype=mask.dtype, device=mask.device)], dim=0)
            padded_embeds.append(piece)
            padded_masks.append(mask)

        self._append_stats(args, original_visual, kept_visual, int(max_len))
        return torch.stack(padded_embeds, dim=0), torch.stack(padded_masks, dim=0), max_len

    def _append_stats(self, args: Any, original_visual: int, kept_visual: int, prompt_tokens: int) -> None:
        save_path = getattr(args, "save_path", None)
        if not save_path:
            return
        try:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "a") as f:
                f.write(json.dumps({
                    "method": "llava_onevision_entitymem",
                    "visual_tokens": int(original_visual),
                    "kept_visual_tokens": int(kept_visual),
                    "kept_tokens": int(prompt_tokens),
                    "short_frames": int(args.short_frames),
                    "medium_frames": int(args.medium_frames),
                    "tokens_per_entity": int(args.entity_tokens_per_entity),
                    "max_entities": int(args.entity_max_entities),
                    "temporal_burst_selection": bool(getattr(args, "entity_temporal_burst_selection", False)),
                    "temporal_burst_max_events": int(getattr(args, "entity_temporal_burst_max_events", 0)),
                    "temporal_burst_radius": int(getattr(args, "entity_temporal_burst_radius", 0)),
                    "temporal_burst_tokens_per_frame": int(getattr(args, "entity_temporal_burst_tokens_per_frame", 0)),
                }) + "\n")
        except Exception:
            pass

    def generate_response(self, messages: list[dict[str, Any]], args: Any, max_new_tokens: int = 128) -> str:
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )
        inputs_embeds, attention_mask, prompt_len = self._prepare_prompt_embeddings(inputs, args)
        with torch.no_grad():
            generated = self.model.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        output_ids = generated[0]
        if output_ids.shape[0] > prompt_len:
            output_ids = output_ids[prompt_len:]
        return self.processor.decode(output_ids, skip_special_tokens=True).strip()
