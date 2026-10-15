from __future__ import annotations

import torch

from objectstream import EntityMemoryConfig
from objectstream.entity_memory import EntityMemoryBank


def test_entity_memory_update_and_select() -> None:
    bank = EntityMemoryBank(
        EntityMemoryConfig(max_entities=4, saliency_quantile=0.5, tokens_per_entity=2)
    )
    features = torch.randn(16, 8)
    saliency = torch.linspace(0, 1, 16)
    updated = bank.update(features, saliency, frame_index=0, grid_size=(4, 4))
    assert updated
    selected = bank.select_tokens(saliency)
    assert selected
    assert len(selected) <= len(updated) * 2


def test_entity_memory_query_retrieval() -> None:
    bank = EntityMemoryBank(EntityMemoryConfig(max_entities=4, saliency_quantile=0.5))
    features = torch.randn(16, 8)
    saliency = torch.linspace(0, 1, 16)
    bank.update(features, saliency, frame_index=0, grid_size=(4, 4))
    query = next(iter(bank.entities.values())).feature.clone()
    retrieved = bank.retrieve_for_query(query, topk=1)
    assert len(retrieved) == 1
