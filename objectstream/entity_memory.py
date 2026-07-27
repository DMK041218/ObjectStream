from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import torch
import torch.nn.functional as F

from .config import EntityMemoryConfig


@dataclass
class EntityState:
    entity_id: int
    feature: torch.Tensor
    saliency: float
    last_seen: int
    age: int = 0
    persistence: float = 1.0
    token_indices: list[int] = field(default_factory=list)


class EntityMemoryBank:
    """Training-free soft entity memory.

    Inputs are visual-token features before the LLM projector when available.
    The module is intentionally independent from Qwen internals so it can be
    tested before patching ObjectStream's model code path.
    """

    def __init__(self, config: EntityMemoryConfig | None = None) -> None:
        self.config = config or EntityMemoryConfig()
        self.entities: dict[int, EntityState] = {}
        self._next_id = 0

    @torch.no_grad()
    def update(
        self,
        token_features: torch.Tensor,
        saliency: torch.Tensor,
        frame_index: int,
        grid_size: tuple[int, int] | None = None,
    ) -> list[EntityState]:
        """Update memory with one frame worth of visual tokens.

        Args:
            token_features: Tensor shaped [num_tokens, dim].
            saliency: Tensor shaped [num_tokens], e.g. aggregated attention.
            frame_index: Monotonic stream frame index.
            grid_size: Optional visual grid [height, width]. If supplied,
                connected components are formed in image space; otherwise
                high-saliency tokens are treated as one soft cluster.
        """

        if token_features.ndim != 2:
            raise ValueError("token_features must be [num_tokens, dim]")
        if saliency.ndim != 1 or saliency.numel() != token_features.shape[0]:
            raise ValueError("saliency must be [num_tokens]")

        token_features = F.normalize(token_features.float(), dim=-1)
        saliency = saliency.float()
        clusters = self._make_clusters(saliency, grid_size)
        updated: list[EntityState] = []
        for token_indices in clusters:
            idx = torch.tensor(token_indices, device=token_features.device)
            cluster_feat = self._weighted_mean(token_features[idx], saliency[idx])
            cluster_saliency = float(saliency[idx].mean().item())
            entity = self._match_or_create(cluster_feat, cluster_saliency, frame_index)
            entity.token_indices = token_indices
            updated.append(entity)

        self._evict_if_needed(frame_index)
        return updated

    @torch.no_grad()
    def select_tokens(
        self,
        saliency: torch.Tensor,
        entity_token_map: dict[int, Iterable[int]] | None = None,
    ) -> list[int]:
        """Select top tokens per entity under a fixed per-entity budget."""

        selected: list[int] = []
        token_map = entity_token_map or {
            entity_id: state.token_indices for entity_id, state in self.entities.items()
        }
        for _, indices in token_map.items():
            idx = list(indices)
            if not idx:
                continue
            scores = saliency[torch.tensor(idx, device=saliency.device)]
            k = min(self.config.tokens_per_entity, len(idx))
            keep = scores.topk(k).indices.tolist()
            selected.extend(idx[i] for i in keep)
        return sorted(set(selected))

    @torch.no_grad()
    def retrieve_for_query(
        self,
        query_feature: torch.Tensor,
        topk: int | None = None,
    ) -> list[EntityState]:
        """Return entities most aligned with a text/query embedding."""

        if not self.entities:
            return []
        topk = topk or self.config.query_topk_entities
        query_feature = F.normalize(query_feature.float(), dim=-1)
        states = list(self.entities.values())
        feats = torch.stack([state.feature.to(query_feature.device) for state in states])
        scores = feats @ query_feature
        keep = scores.topk(min(topk, len(states))).indices.tolist()
        return [states[i] for i in keep]

    def _make_clusters(
        self,
        saliency: torch.Tensor,
        grid_size: tuple[int, int] | None,
    ) -> list[list[int]]:
        threshold = torch.quantile(saliency, self.config.saliency_quantile)
        active = (saliency >= threshold).cpu()
        if grid_size is None:
            indices = active.nonzero(as_tuple=False).flatten().tolist()
            return [indices] if len(indices) >= self.config.min_cluster_size else []

        height, width = grid_size
        if height * width != saliency.numel():
            raise ValueError("grid_size must match num_tokens")

        seen: set[int] = set()
        clusters: list[list[int]] = []
        for start in active.nonzero(as_tuple=False).flatten().tolist():
            if start in seen:
                continue
            stack = [start]
            group: list[int] = []
            seen.add(start)
            while stack:
                current = stack.pop()
                group.append(current)
                row, col = divmod(current, width)
                for nr, nc in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
                    if nr < 0 or nc < 0 or nr >= height or nc >= width:
                        continue
                    nxt = nr * width + nc
                    if nxt not in seen and bool(active[nxt]):
                        seen.add(nxt)
                        stack.append(nxt)
            if len(group) >= self.config.min_cluster_size:
                clusters.append(group)
        return clusters

    def _weighted_mean(self, features: torch.Tensor, saliency: torch.Tensor) -> torch.Tensor:
        weights = saliency.clamp_min(1e-6)
        weights = weights / weights.sum()
        return F.normalize((features * weights[:, None]).sum(dim=0), dim=0)

    def _match_or_create(
        self,
        feature: torch.Tensor,
        saliency: float,
        frame_index: int,
    ) -> EntityState:
        if self.entities:
            ids = list(self.entities.keys())
            feats = torch.stack([self.entities[i].feature.to(feature.device) for i in ids])
            sims = feats @ feature
            best_pos = int(sims.argmax().item())
            best_id = ids[best_pos]
            if float(sims[best_pos].item()) >= self.config.entity_match_threshold:
                state = self.entities[best_id]
                old_feature = state.feature.to(feature.device)
                new_feature = F.normalize(
                    self.config.ema_decay * old_feature
                    + (1.0 - self.config.ema_decay) * feature,
                    dim=0,
                )
                drift = 1.0 - float((old_feature @ feature).item())
                state.persistence = max(0.0, min(1.0, 1.0 - drift))
                state.feature = new_feature.detach().cpu()
                state.saliency = saliency
                state.last_seen = frame_index
                state.age += 1
                return state

        entity_id = self._next_id
        self._next_id += 1
        state = EntityState(
            entity_id=entity_id,
            feature=feature.detach().cpu(),
            saliency=saliency,
            last_seen=frame_index,
        )
        self.entities[entity_id] = state
        return state

    def _evict_if_needed(self, frame_index: int) -> None:
        overflow = len(self.entities) - self.config.max_entities
        if overflow <= 0:
            return
        ranked = sorted(
            self.entities.values(),
            key=lambda s: (s.persistence, s.saliency, -(frame_index - s.last_seen)),
        )
        for state in ranked[:overflow]:
            self.entities.pop(state.entity_id, None)

