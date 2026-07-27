"""Qwen3-VL inference helper with EntityMem/Burst video compression.

For Qwen3-VL we avoid running EntityMem over the full prompt token layout,
because Qwen3's multimodal placeholder scheme differs from the Qwen2.5 path
that EntityMem was originally written against. Instead we:

1. build official Qwen3-VL video embeddings;
2. run EntityMem/Burst on a synthetic visual-only stream;
3. map kept visual indices back to the real Qwen3 prompt;
4. slice prompt embeds / masks / RoPE positions accordingly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from qwen2_5_vl_objectstream.entitymem import EntityMem

try:
    from qwen_vl_utils import process_vision_info
except ModuleNotFoundError:
    from qwen_vl_utils_objectstream import process_vision_info


class Qwen3VLEntityMemGenerator:
    def __init__(self, ckpt_path: str, min_pixels: int, max_pixels: int, device: str = "cuda") -> None:
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            device_map: Any = {"": self.device.index or 0}
        else:
            device_map = "cpu"
        load_kwargs = dict(
            torch_dtype=torch.bfloat16,
            device_map=device_map,
        )
        try:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                ckpt_path,
                attn_implementation="flash_attention_2",
                **load_kwargs,
            ).eval()
            self.attn_impl = "flash_attention_2"
        except Exception:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                ckpt_path,
                attn_implementation="sdpa",
                **load_kwargs,
            ).eval()
            self.attn_impl = "sdpa"
        self.processor = AutoProcessor.from_pretrained(
            ckpt_path,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        self.tokenizer = self.processor.tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _make_entitymem(self, args: Any) -> EntityMem:
        return EntityMem(
            vision_start_token_id=int(self.processor.vision_start_token_id),
            vision_end_token_id=int(self.processor.vision_end_token_id),
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

    @staticmethod
    def _text_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
        text_pos = attention_mask.long().cumsum(-1) - 1
        text_pos = text_pos.masked_fill(attention_mask == 0, 0)
        return text_pos

    def _append_stats(
        self,
        args: Any,
        original_visual: int,
        kept_visual: int,
        kept_prompt_tokens: int,
    ) -> None:
        save_path = getattr(args, "save_path", None)
        if not save_path:
            return
        retention = float(kept_visual) / float(original_visual) if original_visual > 0 else 1.0
        record = {
            "method": "qwen3vl_entitymem",
            "attn_implementation": self.attn_impl,
            "visual_tokens": int(original_visual),
            "kept_visual_tokens": int(kept_visual),
            "retention_ratio": retention,
            "kept_tokens": int(kept_prompt_tokens),
            "short_frames": int(args.short_frames),
            "medium_frames": int(args.medium_frames),
            "tokens_per_entity": int(args.entity_tokens_per_entity),
            "max_entities": int(args.entity_max_entities),
            "temporal_burst_selection": bool(getattr(args, "entity_temporal_burst_selection", False)),
            "temporal_burst_max_events": int(getattr(args, "entity_temporal_burst_max_events", 0)),
            "temporal_burst_radius": int(getattr(args, "entity_temporal_burst_radius", 0)),
            "temporal_burst_tokens_per_frame": int(getattr(args, "entity_temporal_burst_tokens_per_frame", 0)),
        }
        try:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception:
            pass

    @staticmethod
    def _video_grid_shape(video_grid_thw: torch.Tensor) -> tuple[int, int, int]:
        if video_grid_thw.ndim != 2 or video_grid_thw.shape[0] != 1 or video_grid_thw.shape[1] != 3:
            raise ValueError(f"Expected video_grid_thw shape [1, 3], got {tuple(video_grid_thw.shape)}")
        t = int(video_grid_thw[0, 0].item())
        h = int(video_grid_thw[0, 1].item())
        w = int(video_grid_thw[0, 2].item())
        return t, h, w

    def _compress_video_only(
        self,
        video_embeds: torch.Tensor,
        video_pos3: torch.Tensor,
        video_grid_thw: torch.Tensor,
        args: Any,
    ) -> torch.Tensor:
        if video_embeds.ndim != 2:
            raise ValueError(f"Expected video_embeds [N, D], got {tuple(video_embeds.shape)}")
        if video_pos3.ndim != 3 or video_pos3.shape[0] != 3 or video_pos3.shape[1] != 1:
            raise ValueError(f"Expected video_pos3 [3, 1, N], got {tuple(video_pos3.shape)}")

        num_visual, hidden = video_embeds.shape
        start_id = int(self.processor.vision_start_token_id)
        end_id = int(self.processor.vision_end_token_id)
        video_token_id = int(self.processor.video_token_id)

        hidden_states = torch.zeros(
            (1, num_visual + 2, hidden),
            dtype=video_embeds.dtype,
            device=video_embeds.device,
        )
        hidden_states[0, 1:-1] = video_embeds

        input_ids = torch.full(
            (1, num_visual + 2),
            fill_value=video_token_id,
            dtype=torch.long,
            device=video_embeds.device,
        )
        input_ids[0, 0] = start_id
        input_ids[0, -1] = end_id

        pos3 = torch.zeros((3, 1, num_visual + 2), dtype=torch.long, device=video_embeds.device)
        pos3[:, :, 1:-1] = video_pos3
        pos_emb = (
            torch.zeros((1, 1, num_visual + 2, hidden), dtype=video_embeds.dtype, device=video_embeds.device),
            torch.zeros((1, 1, num_visual + 2, hidden), dtype=video_embeds.dtype, device=video_embeds.device),
        )

        entitymem = self._make_entitymem(args)
        entitymem.save_path = None
        _, _, _, _, kept_indices = entitymem.process_memory_streaming(
            hidden_states=hidden_states,
            position_ids=pos3,
            position_embeddings=pos_emb,
            input_ids=input_ids,
            video_grid_thw=video_grid_thw,
        )
        kept = kept_indices[0].to(video_embeds.device)
        kept = kept[(kept > 0) & (kept < num_visual + 1)] - 1
        return kept.sort().values

    def _prepare_prompt_embeddings(self, inputs: Any, args: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        input_ids = inputs.input_ids.to(self.model.device)
        attention_mask = inputs.attention_mask.to(self.model.device)
        mm_token_type_ids = inputs.mm_token_type_ids.to(self.model.device)
        inputs_embeds = self.model.get_input_embeddings()(input_ids)

        pixel_values_videos = getattr(inputs, "pixel_values_videos", None)
        video_grid_thw = getattr(inputs, "video_grid_thw", None)
        if pixel_values_videos is None or video_grid_thw is None or not getattr(args, "use_entitymem", False):
            text_pos = self._text_position_ids(attention_mask)
            pos4 = text_pos.unsqueeze(0).repeat(4, 1, 1)
            return inputs_embeds, attention_mask, pos4, int(inputs_embeds.shape[1])

        pixel_values_videos = pixel_values_videos.to(self.model.device, dtype=torch.bfloat16)
        video_grid_thw = video_grid_thw.to(self.model.device)
        video_outputs = self.model.get_video_features(
            pixel_values_videos,
            video_grid_thw=video_grid_thw,
            return_dict=True,
        )
        video_embeds = torch.cat(video_outputs.pooler_output, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        video_token_id = int(self.processor.video_token_id)
        video_positions = (input_ids[0] == video_token_id).nonzero(as_tuple=True)[0]
        if video_positions.numel() != video_embeds.shape[0]:
            raise ValueError(
                f"Qwen3-VL video placeholder mismatch: prompt has {int(video_positions.numel())} "
                f"video tokens but encoder returned {int(video_embeds.shape[0])} features"
            )

        _, video_mask = self.model.model.get_placeholder_mask(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            video_features=video_embeds,
        )
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        position_ids_3d, _ = self.model.model.get_rope_index(
            input_ids,
            mm_token_type_ids=mm_token_type_ids,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
        )
        if not getattr(args, "use_entitymem", False):
            text_pos = self._text_position_ids(attention_mask)
            pos4 = torch.cat([text_pos.unsqueeze(0), position_ids_3d], dim=0)
            return inputs_embeds, attention_mask, pos4, int(inputs_embeds.shape[1])

        if input_ids.shape[0] != 1:
            raise ValueError("Qwen3-VL EntityMem PoC currently expects batch size 1")

        _, grid_h, grid_w = self._video_grid_shape(video_grid_thw)
        visual_pos3 = position_ids_3d[:, :, video_positions]
        kept_local = self._compress_video_only(
            video_embeds=video_embeds,
            video_pos3=visual_pos3,
            video_grid_thw=torch.tensor([[int(video_grid_thw[0, 0].item()), grid_h, grid_w]], device=video_grid_thw.device),
            args=args,
        )
        kept_video_positions = video_positions[kept_local]

        keep_mask = torch.ones(input_ids.shape[1], dtype=torch.bool, device=input_ids.device)
        keep_mask[video_positions] = False
        keep_mask[kept_video_positions] = True
        keep_positions = keep_mask.nonzero(as_tuple=True)[0]

        compressed_hidden = inputs_embeds[:, keep_positions, :]
        compressed_mask = attention_mask[:, keep_positions]
        compressed_pos3 = position_ids_3d[:, :, keep_positions]

        original_visual = int(video_positions.numel())
        kept_visual = int(kept_video_positions.numel())
        self._append_stats(args, original_visual, kept_visual, int(compressed_hidden.shape[1]))

        text_pos = self._text_position_ids(compressed_mask)
        compressed_pos4 = torch.cat([text_pos.unsqueeze(0), compressed_pos3], dim=0)
        return compressed_hidden, compressed_mask, compressed_pos4, int(compressed_hidden.shape[1])

    def generate_response(self, messages: list[dict[str, Any]], args: Any, max_new_tokens: int = 128) -> str:
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        if isinstance(video_kwargs, dict):
            fps_value = video_kwargs.get("fps")
            if isinstance(fps_value, list) and len(fps_value) == 1:
                video_kwargs["fps"] = float(fps_value[0])
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )
        inputs_embeds, attention_mask, position_ids, prompt_len = self._prepare_prompt_embeddings(inputs, args)
        dummy_input_ids = torch.full(
            (inputs_embeds.shape[0], inputs_embeds.shape[1]),
            fill_value=int(self.tokenizer.pad_token_id),
            dtype=torch.long,
            device=inputs_embeds.device,
        )
        with torch.no_grad():
            generated = self.model.generate(
                input_ids=dummy_input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        output_ids = generated[0]
        if output_ids.shape[0] > prompt_len:
            output_ids = output_ids[prompt_len:]
        return self.processor.decode(output_ids, skip_special_tokens=True).strip()
