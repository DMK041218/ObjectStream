from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .config import EntityMemoryConfig, QwenObjectStreamConfig


def build_messages(video_path: str, question: str, fps: float, max_pixels: int) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"video": video_path, "fps": fps, "max_pixels": max_pixels},
            ],
        },
    ]


def run_qwen25vl_objectstream(
    video_path: str,
    question: str,
    qwen_config: QwenObjectStreamConfig,
    entity_config: EntityMemoryConfig,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run the ObjectStream-patched Qwen2.5-VL API.

    The entity config is returned for traceability. A deeper integration should
    pass EntityMemoryBank into the patched model's visual token path.
    """

    messages = build_messages(
        video_path=video_path,
        question=question,
        fps=qwen_config.fps,
        max_pixels=qwen_config.max_pixels,
    )
    if dry_run:
        return {
            "dry_run": True,
            "messages": messages,
            "qwen_config": asdict(qwen_config),
            "entity_config": asdict(entity_config),
            "note": "No model, data, or video tensors were loaded.",
        }

    import torch

    from qwen_vl_utils_objectstream import process_vision_info
    from qwen2_5_vl_objectstream import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        qwen_config.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=qwen_config.attn_implementation,
        device_map="auto",
        local_files_only=qwen_config.local_files_only,
    )
    processor = Qwen2_5_VLProcessor.from_pretrained(
        qwen_config.model_path,
        local_files_only=qwen_config.local_files_only,
    )

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages, return_video_kwargs=True
    )
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    )
    inputs = inputs.to(model.device)

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=qwen_config.max_new_tokens,
        do_sample=False,
        temperature=0.0,
        # ObjectStream is implemented by the entity-centric memory path in the
        # patched Qwen2.5-VL backend.
        use_objectstream=qwen_config.use_objectstream,
        short_frames=qwen_config.short_frames,
        medium_frames=qwen_config.medium_frames,
        entity_tokens_per_entity=entity_config.tokens_per_entity,
        entity_max_entities=entity_config.max_entities,
        entity_saliency_quantile=entity_config.saliency_quantile,
        entity_match_threshold=entity_config.entity_match_threshold,
        entity_ema_decay=entity_config.ema_decay,
    )
    trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return {
        "dry_run": False,
        "answer": output_text[0] if output_text else "",
        "qwen_config": asdict(qwen_config),
        "entity_config": asdict(entity_config),
    }
