from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EntityMemoryConfig:
    """Training-free entity-memory knobs from the proposal."""

    max_entities: int = 64
    entity_match_threshold: float = 0.72
    ema_decay: float = 0.9
    saliency_quantile: float = 0.75
    min_cluster_size: int = 2
    tokens_per_entity: int = 8
    query_topk_entities: int = 16


@dataclass
class QwenObjectStreamConfig:
    """ObjectStream/Qwen2.5-VL inference knobs."""

    model_path: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    fps: float = 1.0
    max_pixels: int = 256 * 28 * 28
    max_new_tokens: int = 128
    short_frames: int = 8
    medium_frames: int = 64
    use_objectstream: bool = True
    local_files_only: bool = False
    attn_implementation: str = "flash_attention_2"

