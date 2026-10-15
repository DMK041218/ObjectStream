"""Entity-centric streaming memory for Qwen2.5-VL.

Training-free prototype for the RP_cited proposal:
- build soft spatial entities from pre-LLM visual token embeddings;
- keep EMA entity states across frames;
- retain recent frames fully and older frames with per-entity token budgets;
- expose the same return contract as ObjectStream.process_memory_streaming.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from .utils import scan_visual_indices, right_pad_and_stack


@dataclass
class _EntityState:
    entity_id: int
    feature: torch.Tensor
    last_seen: int
    saliency: float
    persistence: float = 1.0
    age: int = 0


class EntityMem:
    def __init__(
        self,
        vision_start_token_id: int,
        vision_end_token_id: int,
        short_term_frames: int = 8,
        mid_term_frames: int = 64,
        max_entities: int = 64,
        tokens_per_entity: int = 8,
        saliency_quantile: float = 0.75,
        match_threshold: float = 0.72,
        ema_decay: float = 0.9,
        min_cluster_size: int = 2,
        temporal_selection: bool = False,
        temporal_boundary_selection: bool = False,
        temporal_even_selection: bool = False,
        temporal_novelty_selection: bool = False,
        temporal_gap_reset: bool = False,
        temporal_gap_reset_frames: Optional[int] = None,
        temporal_anchor_selection: bool = False,
        temporal_anchor_bins: int = 8,
        temporal_anchor_tokens_per_bin: int = 4,
        temporal_keyframe_selection: bool = False,
        temporal_keyframe_max_frames: int = 16,
        temporal_keyframe_tokens_per_frame: int = 2,
        temporal_keyframe_grid_selection: bool = False,
        temporal_keyframe_grid_size: int = 4,
        temporal_event_selection: bool = False,
        temporal_event_max_frames: int = 64,
        temporal_event_tokens_per_frame: int = 8,
        temporal_burst_selection: bool = False,
        temporal_burst_max_events: int = 32,
        temporal_burst_max_events_per_entity: int = 0,
        temporal_burst_radius: int = 2,
        temporal_burst_tokens_per_frame: int = 8,
        chunk_selection: bool = False,
        chunk_size_frames: int = 64,
        chunk_max_entities: int = 0,
        temporal_recap_selection: bool = False,
        temporal_recap_bins: int = 32,
        temporal_recap_tokens_per_bin: int = 4,
        temporal_content_encoding: bool = False,
        temporal_content_scale: float = 0.02,
        temporal_order_content_encoding: bool = False,
        temporal_order_content_scale: float = 0.02,
        query_saliency: bool = False,
        query_saliency_mix: float = 0.35,
    ) -> None:
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        self.short_term_frames = int(short_term_frames)
        self.mid_term_frames = int(mid_term_frames)
        self.temporal_patch_size = 2
        self.max_entities = int(max_entities)
        self.tokens_per_entity = int(tokens_per_entity)
        self.saliency_quantile = float(saliency_quantile)
        self.match_threshold = float(match_threshold)
        self.ema_decay = float(ema_decay)
        self.min_cluster_size = int(min_cluster_size)
        self.temporal_selection = bool(temporal_selection)
        self.temporal_boundary_selection = bool(temporal_boundary_selection)
        self.temporal_even_selection = bool(temporal_even_selection)
        self.temporal_novelty_selection = bool(temporal_novelty_selection)
        self.temporal_gap_reset = bool(temporal_gap_reset)
        self.temporal_gap_reset_frames = None if temporal_gap_reset_frames is None else int(temporal_gap_reset_frames)
        self.temporal_anchor_selection = bool(temporal_anchor_selection)
        self.temporal_anchor_bins = int(temporal_anchor_bins)
        self.temporal_anchor_tokens_per_bin = int(temporal_anchor_tokens_per_bin)
        self.temporal_keyframe_selection = bool(temporal_keyframe_selection)
        self.temporal_keyframe_max_frames = int(temporal_keyframe_max_frames)
        self.temporal_keyframe_tokens_per_frame = int(temporal_keyframe_tokens_per_frame)
        self.temporal_keyframe_grid_selection = bool(temporal_keyframe_grid_selection)
        self.temporal_keyframe_grid_size = int(temporal_keyframe_grid_size)
        self.temporal_event_selection = bool(temporal_event_selection)
        self.temporal_event_max_frames = int(temporal_event_max_frames)
        self.temporal_event_tokens_per_frame = int(temporal_event_tokens_per_frame)
        self.temporal_burst_selection = bool(temporal_burst_selection)
        self.temporal_burst_max_events = int(temporal_burst_max_events)
        self.temporal_burst_max_events_per_entity = int(temporal_burst_max_events_per_entity)
        self.temporal_burst_radius = int(temporal_burst_radius)
        self.temporal_burst_tokens_per_frame = int(temporal_burst_tokens_per_frame)
        self.chunk_selection = bool(chunk_selection)
        self.chunk_size_frames = int(chunk_size_frames)
        self.chunk_max_entities = int(chunk_max_entities)
        self.temporal_recap_selection = bool(temporal_recap_selection)
        self.temporal_recap_bins = int(temporal_recap_bins)
        self.temporal_recap_tokens_per_bin = int(temporal_recap_tokens_per_bin)
        self.temporal_content_encoding = bool(temporal_content_encoding)
        self.temporal_content_scale = float(temporal_content_scale)
        self.temporal_order_content_encoding = bool(temporal_order_content_encoding)
        self.temporal_order_content_scale = float(temporal_order_content_scale)
        self.query_saliency = bool(query_saliency)
        self.query_saliency_mix = float(query_saliency_mix)
        self.save_path: str | None = None
        self._next_id = 0

    @staticmethod
    def _pack_sample(hidden_states, position_ids, position_embeddings, batch_index, kept_indices):
        hidden_idx = kept_indices.to(hidden_states.device)
        pos_idx = kept_indices.to(position_ids.device)
        pos1_idx = kept_indices.to(position_embeddings[0].device)
        pos2_idx = kept_indices.to(position_embeddings[1].device)
        return {
            "hidden": hidden_states[batch_index, hidden_idx],
            "pos_ids": position_ids[:, batch_index, pos_idx],
            "pos_emb1": position_embeddings[0][:, batch_index, pos1_idx],
            "pos_emb2": position_embeddings[1][:, batch_index, pos2_idx],
        }

    @staticmethod
    def _localize_spatial_ids(visual_indices, height_ids, width_ids):
        if visual_indices.numel() == 0:
            return height_ids, width_ids
        height_ids = height_ids.clone() - height_ids[visual_indices].min()
        width_ids = width_ids.clone() - width_ids[visual_indices].min()
        return height_ids, width_ids

    @staticmethod
    def _compute_grid_hw(visual_indices, height_ids, width_ids):
        if visual_indices.numel() == 0:
            return None
        return int(height_ids[visual_indices].max().item()) + 1, int(width_ids[visual_indices].max().item()) + 1

    @staticmethod
    def _weighted_mean(features: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        weights = weights.float().clamp_min(1e-6)
        weights = weights / weights.sum().clamp_min(1e-6)
        return F.normalize((features.float() * weights[:, None]).sum(dim=0), dim=0, eps=1e-8)

    def _frame_maps(self, visual_indices, time_ids, height_ids, width_ids, grid_hw):
        time_vals = time_ids[visual_indices].detach().cpu()
        frames = [int(x) for x in torch.unique(time_vals, sorted=True).tolist()]
        frame_to_indices = {}
        frame_to_grid = {}
        for frame_id in frames:
            idx = visual_indices[time_vals == frame_id]
            frame_to_indices[frame_id] = idx
            if grid_hw is not None:
                grid = torch.full(grid_hw, -1, dtype=torch.long)
                h_cpu = height_ids[idx].detach().cpu().long()
                w_cpu = width_ids[idx].detach().cpu().long()
                grid[h_cpu, w_cpu] = idx.long()
                frame_to_grid[frame_id] = grid
        return frames, frame_to_indices, frame_to_grid

    def _connected_components(self, frame_indices, saliency, height_ids, width_ids, grid, device):
        if frame_indices.numel() == 0:
            return []
        frame_scores = saliency[frame_indices].detach().cpu().float()
        threshold = torch.quantile(frame_scores, self.saliency_quantile)
        active_global = frame_indices[frame_scores >= threshold]
        active = set(int(x) for x in active_global.detach().cpu().tolist())
        seen = set()
        clusters: list[list[int]] = []
        for start in list(active):
            if start in seen:
                continue
            stack = [start]
            seen.add(start)
            group = []
            while stack:
                cur = stack.pop()
                group.append(cur)
                h = int(height_ids[cur].item())
                w = int(width_ids[cur].item())
                for dh, dw in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nh = h + dh
                    nw = w + dw
                    if nh < 0 or nw < 0 or nh >= grid.shape[0] or nw >= grid.shape[1]:
                        continue
                    nxt = int(grid[nh, nw].item())
                    if nxt >= 0 and nxt in active and nxt not in seen:
                        seen.add(nxt)
                        stack.append(nxt)
            if len(group) >= self.min_cluster_size:
                clusters.append(group)
        if not clusters:
            k = min(max(self.tokens_per_entity, self.min_cluster_size), int(frame_indices.numel()))
            top_local = saliency[frame_indices].detach().cpu().topk(k).indices
            clusters.append([int(x) for x in frame_indices[top_local].detach().cpu().tolist()])
        return clusters

    def _temporal_gap_limit(self) -> int:
        if self.temporal_gap_reset_frames is not None:
            return max(1, int(self.temporal_gap_reset_frames))
        # Work in post-patch temporal ids. A quarter of the mid-term window keeps
        # continuous tracks together while separating repeated long-gap episodes.
        derived = int(self.mid_term_frames / max(1, int(self.temporal_patch_size)) / 4)
        return max(8, derived)

    def _match_or_create(self, states: dict[int, _EntityState], feature: torch.Tensor, saliency: float, frame_id: int) -> int:
        if states:
            if self.temporal_gap_reset:
                gap_limit = self._temporal_gap_limit()
                ids = [eid for eid, st in states.items() if int(frame_id) - int(st.last_seen) <= gap_limit]
            else:
                ids = list(states.keys())
            if ids:
                feats = torch.stack([states[i].feature.to(feature.device) for i in ids], dim=0)
                sims = feats @ feature
                best_pos = int(torch.argmax(sims).item())
                best_id = ids[best_pos]
                if float(sims[best_pos].item()) >= self.match_threshold:
                    st = states[best_id]
                    old = st.feature.to(feature.device)
                    drift = 1.0 - float((old @ feature).item())
                    st.feature = F.normalize(self.ema_decay * old + (1.0 - self.ema_decay) * feature, dim=0, eps=1e-8).detach().cpu()
                    st.persistence = max(0.0, min(1.0, 1.0 - drift))
                    st.saliency = float(saliency)
                    st.last_seen = int(frame_id)
                    st.age += 1
                    return best_id
        eid = self._next_id
        self._next_id += 1
        states[eid] = _EntityState(eid, feature.detach().cpu(), int(frame_id), float(saliency))
        return eid

    def _evict_states(self, states: dict[int, _EntityState], current_frame: int) -> None:
        overflow = len(states) - self.max_entities
        if overflow <= 0:
            return
        ranked = sorted(states.values(), key=lambda s: (s.persistence, s.saliency, -(current_frame - s.last_seen)))
        for st in ranked[:overflow]:
            states.pop(st.entity_id, None)

    @staticmethod
    def _select_temporal_tokens(
        tidx: torch.Tensor,
        scores: torch.Tensor,
        frame_ids: list[int],
        k: int,
        boundary_first: bool = False,
        even_coverage: bool = False,
        novelty_coverage: bool = False,
    ) -> torch.Tensor:
        """Pick salient entity tokens while preserving temporal coverage.

        Counting and ordering questions are brittle when all per-entity slots come
        from one high-saliency moment. The optional boundary-first mode explicitly
        keeps the first and last seen moments for each entity. Novelty coverage then
        reserves slots for frames whose entity saliency changes sharply over time.
        """
        if k <= 0 or tidx.numel() == 0:
            return tidx[:0]
        k = min(int(k), int(tidx.numel()))
        score_cpu = scores.detach().cpu().float()
        frame_to_local: dict[int, list[int]] = {}
        for local_idx, frame_id in enumerate(frame_ids):
            frame_to_local.setdefault(int(frame_id), []).append(local_idx)

        chosen_local: list[int] = []
        used: set[int] = set()

        def add_best_for_frame(frame_id: int) -> None:
            if len(chosen_local) >= k:
                return
            candidates = [i for i in frame_to_local.get(int(frame_id), []) if i not in used]
            if not candidates:
                return
            best = max(candidates, key=lambda i: float(score_cpu[i].item()))
            chosen_local.append(best)
            used.add(best)

        sorted_frames = sorted(frame_to_local)
        if boundary_first and sorted_frames:
            add_best_for_frame(sorted_frames[0])
            if len(sorted_frames) > 1:
                add_best_for_frame(sorted_frames[-1])

        if novelty_coverage and len(sorted_frames) > 2 and len(chosen_local) < k:
            frame_score = {}
            for frame_id, local_idxs in frame_to_local.items():
                vals = [float(score_cpu[i].item()) for i in local_idxs]
                frame_score[int(frame_id)] = max(vals) if vals else 0.0
            novelty = []
            for pos, frame_id in enumerate(sorted_frames):
                prev_score = frame_score[sorted_frames[pos - 1]] if pos > 0 else frame_score[frame_id]
                next_score = frame_score[sorted_frames[pos + 1]] if pos + 1 < len(sorted_frames) else frame_score[frame_id]
                cur_score = frame_score[frame_id]
                novelty.append((abs(cur_score - prev_score) + abs(cur_score - next_score), frame_id))
            # Keep a few change points before filling the remaining slots by coverage/saliency.
            novelty_budget = max(1, min(k - len(chosen_local), len(sorted_frames) // 4))
            for _, frame_id in sorted(novelty, reverse=True)[:novelty_budget]:
                add_best_for_frame(frame_id)
                if len(chosen_local) >= k:
                    break

        coverage_frames = sorted_frames
        if even_coverage and len(sorted_frames) > 1:
            slots = max(0, k - len(chosen_local))
            if slots == 1:
                coverage_frames = [sorted_frames[len(sorted_frames) // 2]]
            elif slots > 1:
                positions = torch.linspace(0, len(sorted_frames) - 1, steps=min(slots, len(sorted_frames)))
                coverage_frames = [sorted_frames[int(round(float(pos.item())))] for pos in positions]

        for frame_id in coverage_frames:
            add_best_for_frame(frame_id)
            if len(chosen_local) >= k:
                break

        if len(chosen_local) < k:
            remaining = [i for i in range(int(tidx.numel())) if i not in used]
            remaining.sort(key=lambda i: float(score_cpu[i].item()), reverse=True)
            chosen_local.extend(remaining[: k - len(chosen_local)])

        # Restore sequence order so RoPE/time ids remain monotonic in the packed stream.
        chosen = tidx[torch.tensor(chosen_local, dtype=torch.long)]
        return chosen.sort().values

    def _select_temporal_anchor_tokens(
        self,
        old_frames: list[int],
        frame_to_indices: dict[int, torch.Tensor],
        saliency: torch.Tensor,
        already_kept: torch.Tensor,
    ) -> torch.Tensor:
        if (
            not self.temporal_anchor_selection
            or self.temporal_anchor_bins <= 0
            or self.temporal_anchor_tokens_per_bin <= 0
            or not old_frames
        ):
            return torch.empty(0, dtype=torch.long)

        frames = sorted(int(x) for x in old_frames)
        bins = min(max(1, int(self.temporal_anchor_bins)), len(frames))
        positions = torch.linspace(0, len(frames), steps=bins + 1).round().long().tolist()
        chosen: list[int] = []
        kept_cpu = already_kept.detach().cpu()

        for b in range(bins):
            start = int(positions[b])
            end = int(positions[b + 1])
            if end <= start:
                end = min(len(frames), start + 1)
            bin_frames = frames[start:end]
            if not bin_frames:
                continue
            candidates = []
            for frame_id in bin_frames:
                idx = frame_to_indices.get(frame_id)
                if idx is None or idx.numel() == 0:
                    continue
                idx_cpu = idx.detach().cpu().long()
                idx_cpu = idx_cpu[~kept_cpu[idx_cpu]]
                if idx_cpu.numel() > 0:
                    candidates.append(idx_cpu)
            if not candidates:
                continue
            cand = torch.cat(candidates, dim=0)
            k = min(int(self.temporal_anchor_tokens_per_bin), int(cand.numel()))
            scores = saliency[cand.to(saliency.device)].detach().cpu().float()
            top = scores.topk(k).indices
            chosen.extend(int(x) for x in cand[top].tolist())

        if not chosen:
            return torch.empty(0, dtype=torch.long)
        return torch.tensor(sorted(set(chosen)), dtype=torch.long)

    def _select_temporal_keyframe_tokens(
        self,
        old_frames: list[int],
        frame_to_indices: dict[int, torch.Tensor],
        saliency: torch.Tensor,
        already_kept: torch.Tensor,
        height_ids: Optional[torch.Tensor] = None,
        width_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Keep sparse high-saliency tokens from explicit time keyframes.

        Entity matching can merge repeated objects across distant events. These
        keyframes preserve a coarse temporal spine with original RoPE time ids,
        which helps counting/order questions without turning old frames back on.
        """
        if (
            not self.temporal_keyframe_selection
            or self.temporal_keyframe_max_frames <= 0
            or self.temporal_keyframe_tokens_per_frame <= 0
            or not old_frames
        ):
            return torch.empty(0, dtype=torch.long)

        frames = sorted(int(x) for x in old_frames)
        max_frames = min(int(self.temporal_keyframe_max_frames), len(frames))
        if max_frames <= 0:
            return torch.empty(0, dtype=torch.long)

        frame_scores: dict[int, float] = {}
        for frame_id in frames:
            idx = frame_to_indices.get(frame_id)
            if idx is None or idx.numel() == 0:
                frame_scores[frame_id] = 0.0
            else:
                vals = saliency[idx.to(saliency.device)].detach().cpu().float()
                frame_scores[frame_id] = float(vals.max().item()) if vals.numel() else 0.0

        chosen_frames: set[int] = set()
        coverage_slots = max(1, max_frames // 2)
        if len(frames) == 1:
            chosen_frames.add(frames[0])
        else:
            positions = torch.linspace(0, len(frames) - 1, steps=min(coverage_slots, len(frames)))
            chosen_frames.update(frames[int(round(float(pos.item())))] for pos in positions)

        novelty: list[tuple[float, int]] = []
        for pos, frame_id in enumerate(frames):
            prev_score = frame_scores[frames[pos - 1]] if pos > 0 else frame_scores[frame_id]
            next_score = frame_scores[frames[pos + 1]] if pos + 1 < len(frames) else frame_scores[frame_id]
            cur_score = frame_scores[frame_id]
            change = abs(cur_score - prev_score) + abs(cur_score - next_score)
            novelty.append((change + 0.25 * cur_score, frame_id))
        for _, frame_id in sorted(novelty, reverse=True):
            if len(chosen_frames) >= max_frames:
                break
            chosen_frames.add(frame_id)

        kept_cpu = already_kept.detach().cpu()
        chosen_tokens: list[int] = []
        for frame_id in sorted(chosen_frames):
            idx = frame_to_indices.get(frame_id)
            if idx is None or idx.numel() == 0:
                continue
            idx_cpu = idx.detach().cpu().long()
            idx_cpu = idx_cpu[~kept_cpu[idx_cpu]]
            if idx_cpu.numel() == 0:
                continue
            k = min(int(self.temporal_keyframe_tokens_per_frame), int(idx_cpu.numel()))
            scores = saliency[idx_cpu.to(saliency.device)].detach().cpu().float()
            if (
                self.temporal_keyframe_grid_selection
                and height_ids is not None
                and width_ids is not None
                and self.temporal_keyframe_grid_size > 1
            ):
                grid = max(1, int(self.temporal_keyframe_grid_size))
                h = height_ids[idx_cpu.to(height_ids.device)].detach().cpu().long()
                w = width_ids[idx_cpu.to(width_ids.device)].detach().cpu().long()
                h_span = int(h.max().item() - h.min().item() + 1)
                w_span = int(w.max().item() - w.min().item() + 1)
                h_bin = ((h - h.min()).float() * grid / max(1, h_span)).floor().clamp(0, grid - 1).long()
                w_bin = ((w - w.min()).float() * grid / max(1, w_span)).floor().clamp(0, grid - 1).long()
                cell_to_local: dict[tuple[int, int], list[int]] = {}
                for local_idx, cell in enumerate(zip(h_bin.tolist(), w_bin.tolist())):
                    cell_to_local.setdefault((int(cell[0]), int(cell[1])), []).append(local_idx)
                cell_best = []
                for cell, local_idxs in cell_to_local.items():
                    best = max(local_idxs, key=lambda i: float(scores[i].item()))
                    cell_best.append((float(scores[best].item()), best))
                cell_best.sort(reverse=True)
                chosen_local: list[int] = [best for _, best in cell_best[:k]]
                if len(chosen_local) < k:
                    used = set(chosen_local)
                    rest = [i for i in range(int(idx_cpu.numel())) if i not in used]
                    rest.sort(key=lambda i: float(scores[i].item()), reverse=True)
                    chosen_local.extend(rest[: k - len(chosen_local)])
                chosen_tokens.extend(int(x) for x in idx_cpu[torch.tensor(chosen_local, dtype=torch.long)].tolist())
            else:
                top = scores.topk(k).indices
                chosen_tokens.extend(int(x) for x in idx_cpu[top].tolist())

        if not chosen_tokens:
            return torch.empty(0, dtype=torch.long)
        return torch.tensor(sorted(set(chosen_tokens)), dtype=torch.long)

    def _select_temporal_event_tokens(
        self,
        old_frames: list[int],
        frame_to_indices: dict[int, torch.Tensor],
        saliency: torch.Tensor,
        hidden_norm: torch.Tensor,
        already_kept: torch.Tensor,
    ) -> torch.Tensor:
        """Keep tokens from explicit temporal change points.

        Unlike keyframe selection, which mostly follows per-token saliency,
        event selection scores a frame by how much its whole visual state differs
        from neighboring frames. This stores coarse "when the scene/event changed"
        evidence while preserving original RoPE time ids for the selected tokens.
        """
        if (
            not self.temporal_event_selection
            or self.temporal_event_max_frames <= 0
            or self.temporal_event_tokens_per_frame <= 0
            or not old_frames
        ):
            return torch.empty(0, dtype=torch.long)

        frames = sorted(int(x) for x in old_frames)
        means: dict[int, torch.Tensor] = {}
        frame_saliency: dict[int, float] = {}
        for frame_id in frames:
            idx = frame_to_indices.get(frame_id)
            if idx is None or idx.numel() == 0:
                continue
            feats = hidden_norm[idx.to(hidden_norm.device)].float()
            means[frame_id] = F.normalize(feats.mean(dim=0), dim=0, eps=1e-8).detach().cpu()
            vals = saliency[idx.to(saliency.device)].detach().cpu().float()
            frame_saliency[frame_id] = float(vals.max().item()) if vals.numel() else 0.0
        if not means:
            return torch.empty(0, dtype=torch.long)

        event_scores: list[tuple[float, int]] = []
        available = [f for f in frames if f in means]
        for pos, frame_id in enumerate(available):
            cur = means[frame_id]
            prev_score = 0.0
            next_score = 0.0
            if pos > 0:
                prev_score = 1.0 - float((cur @ means[available[pos - 1]]).item())
            if pos + 1 < len(available):
                next_score = 1.0 - float((cur @ means[available[pos + 1]]).item())
            change = max(prev_score, next_score)
            score = change + 0.15 * frame_saliency.get(frame_id, 0.0)
            event_scores.append((score, frame_id))

        max_frames = min(int(self.temporal_event_max_frames), len(event_scores))
        if max_frames <= 0:
            return torch.empty(0, dtype=torch.long)
        chosen_frames = {frame_id for _, frame_id in sorted(event_scores, reverse=True)[:max_frames]}

        kept_cpu = already_kept.detach().cpu()
        chosen_tokens: list[int] = []
        for frame_id in sorted(chosen_frames):
            idx = frame_to_indices.get(frame_id)
            if idx is None or idx.numel() == 0:
                continue
            idx_cpu = idx.detach().cpu().long()
            idx_cpu = idx_cpu[~kept_cpu[idx_cpu]]
            if idx_cpu.numel() == 0:
                continue
            k = min(int(self.temporal_event_tokens_per_frame), int(idx_cpu.numel()))
            scores = saliency[idx_cpu.to(saliency.device)].detach().cpu().float()
            top = scores.topk(k).indices
            chosen_tokens.extend(int(x) for x in idx_cpu[top].tolist())

        if not chosen_tokens:
            return torch.empty(0, dtype=torch.long)
        return torch.tensor(sorted(set(chosen_tokens)), dtype=torch.long)

    def _select_temporal_burst_tokens(
        self,
        entity_tokens: dict[int, list[int]],
        entity_times: dict[int, list[int]],
        saliency: torch.Tensor,
        hidden_norm: torch.Tensor,
        already_kept: torch.Tensor,
    ) -> torch.Tensor:
        """Keep entity-local token bursts around high-change frames.

        This bypasses entity-internal top-k selection for repeated events. For
        each entity, we build a per-frame region feature from that entity's
        original tokens, find frames whose region feature changes unusually, and
        force the entity tokens around those frames into the final keep mask.
        """
        if (
            not self.temporal_burst_selection
            or self.temporal_burst_max_events <= 0
            or self.temporal_burst_radius < 0
            or self.temporal_burst_tokens_per_frame <= 0
            or not entity_tokens
        ):
            return torch.empty(0, dtype=torch.long)

        kept_cpu = already_kept.detach().cpu()
        candidates: list[tuple[float, int, int, torch.Tensor]] = []

        for eid, toks in entity_tokens.items():
            times = entity_times.get(eid, [])
            if not toks or not times or len(toks) != len(times):
                continue

            frame_to_entity_tokens: dict[int, list[int]] = {}
            for tok, frame_id in zip(toks, times):
                frame_to_entity_tokens.setdefault(int(frame_id), []).append(int(tok))
            frames = sorted(frame_to_entity_tokens)
            if len(frames) < 2:
                continue

            means: dict[int, torch.Tensor] = {}
            frame_saliency: dict[int, float] = {}
            for frame_id in frames:
                idx_cpu = torch.tensor(frame_to_entity_tokens[frame_id], dtype=torch.long)
                if idx_cpu.numel() == 0:
                    continue
                idx_dev = idx_cpu.to(hidden_norm.device)
                feats = hidden_norm[idx_dev].float()
                means[frame_id] = F.normalize(feats.mean(dim=0), dim=0, eps=1e-8).detach().cpu()
                vals = saliency[idx_cpu.to(saliency.device)].detach().cpu().float()
                frame_saliency[frame_id] = float(vals.max().item()) if vals.numel() else 0.0
            available = [f for f in frames if f in means]
            if len(available) < 2:
                continue

            changes: list[tuple[int, float]] = []
            for pos, frame_id in enumerate(available):
                cur = means[frame_id]
                prev_score = 0.0
                next_score = 0.0
                if pos > 0:
                    prev_score = 1.0 - float((cur @ means[available[pos - 1]]).item())
                if pos + 1 < len(available):
                    next_score = 1.0 - float((cur @ means[available[pos + 1]]).item())
                score = max(prev_score, next_score) + 0.15 * frame_saliency.get(frame_id, 0.0)
                changes.append((frame_id, score))

            if not changes:
                continue
            vals = torch.tensor([score for _, score in changes], dtype=torch.float32)
            threshold = float((vals.mean() + vals.std(unbiased=False)).item()) if vals.numel() > 1 else float(vals.mean().item())
            radius = max(0, int(self.temporal_burst_radius))
            available_pos = {frame_id: pos for pos, frame_id in enumerate(available)}

            for frame_id, score in changes:
                if score < threshold:
                    continue
                pos = available_pos.get(frame_id)
                if pos is None:
                    continue
                chosen_tokens: list[int] = []
                lo = max(0, pos - radius)
                hi = min(len(available) - 1, pos + radius)
                for burst_pos in range(lo, hi + 1):
                    burst_frame = available[burst_pos]
                    chosen_tokens.extend(frame_to_entity_tokens.get(burst_frame, []))
                if not chosen_tokens:
                    continue
                idx_cpu = torch.tensor(sorted(set(chosen_tokens)), dtype=torch.long)
                idx_cpu = idx_cpu[~kept_cpu[idx_cpu]]
                if idx_cpu.numel() == 0:
                    continue
                candidates.append((float(score), int(eid), int(frame_id), idx_cpu))

        per_entity = max(0, int(getattr(self, "temporal_burst_max_events_per_entity", 0)))
        if per_entity > 0:
            by_entity: dict[int, list[tuple[float, int, int, torch.Tensor]]] = {}
            for candidate in candidates:
                by_entity.setdefault(candidate[1], []).append(candidate)
            balanced_candidates: list[tuple[float, int, int, torch.Tensor]] = []
            for entity_candidates in by_entity.values():
                balanced_candidates.extend(sorted(entity_candidates, reverse=True)[:per_entity])
            candidates = balanced_candidates

        max_events = min(int(self.temporal_burst_max_events), len(candidates))
        if max_events <= 0:
            return torch.empty(0, dtype=torch.long)

        chosen_tokens: list[int] = []
        for _, _, _, idx_cpu in sorted(candidates, reverse=True)[:max_events]:
            k = min(int(self.temporal_burst_tokens_per_frame), int(idx_cpu.numel()))
            scores = saliency[idx_cpu.to(saliency.device)].detach().cpu().float()
            top = scores.topk(k).indices
            chosen_tokens.extend(int(x) for x in idx_cpu[top].tolist())

        if not chosen_tokens:
            return torch.empty(0, dtype=torch.long)
        return torch.tensor(sorted(set(chosen_tokens)), dtype=torch.long)

    def _select_temporal_recap_tokens(
        self,
        old_frames: list[int],
        frame_to_indices: dict[int, torch.Tensor],
        saliency: torch.Tensor,
        hidden_norm: torch.Tensor,
        already_kept: torch.Tensor,
    ) -> torch.Tensor:
        """Keep representative recap tokens for coarse time segments.

        Event tokens focus on sharp changes. Recap tokens are smoother: each time
        bin keeps tokens closest to the bin-level visual mean, with a small saliency
        tie-breaker. This stores "what was generally happening then" without using
        the query text.
        """
        if (
            not self.temporal_recap_selection
            or self.temporal_recap_bins <= 0
            or self.temporal_recap_tokens_per_bin <= 0
            or not old_frames
        ):
            return torch.empty(0, dtype=torch.long)

        frames = sorted(int(x) for x in old_frames)
        bins = min(max(1, int(self.temporal_recap_bins)), len(frames))
        positions = torch.linspace(0, len(frames), steps=bins + 1).round().long().tolist()
        kept_cpu = already_kept.detach().cpu()
        chosen_tokens: list[int] = []

        for b in range(bins):
            start = int(positions[b])
            end = int(positions[b + 1])
            if end <= start:
                end = min(len(frames), start + 1)
            bin_frames = frames[start:end]
            candidates = []
            for frame_id in bin_frames:
                idx = frame_to_indices.get(frame_id)
                if idx is None or idx.numel() == 0:
                    continue
                idx_cpu = idx.detach().cpu().long()
                idx_cpu = idx_cpu[~kept_cpu[idx_cpu]]
                if idx_cpu.numel() > 0:
                    candidates.append(idx_cpu)
            if not candidates:
                continue

            cand = torch.cat(candidates, dim=0)
            cand_dev = cand.to(hidden_norm.device)
            feats = hidden_norm[cand_dev].float()
            center = F.normalize(feats.mean(dim=0), dim=0, eps=1e-8)
            represent = (feats @ center).detach().cpu().float()
            sal = saliency[cand.to(saliency.device)].detach().cpu().float()
            if sal.numel() > 0:
                sal = (sal - sal.min()) / (sal.max() - sal.min()).clamp_min(1e-6)
            score = represent + 0.20 * sal
            k = min(int(self.temporal_recap_tokens_per_bin), int(cand.numel()))
            top = score.topk(k).indices
            chosen_tokens.extend(int(x) for x in cand[top].tolist())

        if not chosen_tokens:
            return torch.empty(0, dtype=torch.long)
        return torch.tensor(sorted(set(chosen_tokens)), dtype=torch.long)

    @staticmethod
    def _build_temporal_content_encoding(time_values: torch.Tensor, hidden_dim: int) -> torch.Tensor:
        """Deterministic sin/cos time code added to selected memory tokens."""
        if time_values.numel() == 0 or hidden_dim <= 0:
            return torch.empty((int(time_values.numel()), hidden_dim), device=time_values.device, dtype=torch.float32)
        t = time_values.float()
        span = (t.max() - t.min()).clamp_min(1.0)
        t = (t - t.min()) / span
        half = max(1, hidden_dim // 2)
        freq = torch.exp(
            torch.arange(half, device=time_values.device, dtype=torch.float32)
            * (-torch.log(torch.tensor(10000.0, device=time_values.device)) / max(1, half - 1))
        )
        angles = t[:, None] * freq[None, :] * (2.0 * torch.pi)
        code = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
        if code.shape[1] < hidden_dim:
            code = F.pad(code, (0, hidden_dim - code.shape[1]))
        return code[:, :hidden_dim]

    @staticmethod
    def _temporal_order_values(time_values: torch.Tensor) -> torch.Tensor:
        if time_values.numel() == 0:
            return time_values.float()
        unique = torch.unique(time_values.detach().float().cpu(), sorted=True)
        rank = {float(v.item()): i for i, v in enumerate(unique)}
        vals = [rank[float(v.item())] for v in time_values.detach().float().cpu()]
        return torch.tensor(vals, device=time_values.device, dtype=torch.float32)

    def _apply_temporal_content_encoding(
        self,
        sample_hidden: torch.Tensor,
        selected_indices: torch.Tensor,
        time_ids: torch.Tensor,
    ) -> torch.Tensor:
        if (
            (not self.temporal_content_encoding and not self.temporal_order_content_encoding)
            or selected_indices.numel() == 0
        ):
            return sample_hidden
        selected = selected_indices.to(sample_hidden.device)
        time_values = time_ids[selected.to(time_ids.device)].to(sample_hidden.device)
        out = sample_hidden.clone()
        delta = torch.zeros_like(out[selected])
        if self.temporal_content_encoding and self.temporal_content_scale > 0:
            code = self._build_temporal_content_encoding(time_values, sample_hidden.shape[-1]).to(
                device=sample_hidden.device,
                dtype=sample_hidden.dtype,
            )
            delta = delta + float(self.temporal_content_scale) * code
        if self.temporal_order_content_encoding and self.temporal_order_content_scale > 0:
            order_values = self._temporal_order_values(time_values).to(sample_hidden.device)
            code = self._build_temporal_content_encoding(order_values, sample_hidden.shape[-1]).to(
                device=sample_hidden.device,
                dtype=sample_hidden.dtype,
            )
            delta = delta + float(self.temporal_order_content_scale) * code
        out[selected] = out[selected] + delta
        return out

    def _mix_query_saliency(
        self,
        saliency: torch.Tensor,
        hidden_norm: torch.Tensor,
        sample_input_ids: torch.Tensor,
        visual_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Blend query-relevant token similarity into the visual saliency score.

        The original entity memory is query-agnostic, which can preserve visually
        salient but irrelevant old evidence. For multiple-choice offline video QA,
        the question/options are already in the prompt, so a light text-visual
        similarity prior helps the temporal spine keep evidence that the question
        is likely to ask about while still respecting temporal coverage.
        """
        if not self.query_saliency or visual_indices.numel() == 0:
            return saliency
        mix = max(0.0, min(1.0, float(self.query_saliency_mix)))
        if mix <= 0:
            return saliency

        device = hidden_norm.device
        visual_on_device = visual_indices.to(device=device)
        text_mask = torch.ones(sample_input_ids.shape[0], dtype=torch.bool, device=device)
        text_mask[visual_on_device] = False
        input_on_device = sample_input_ids.to(device=device)
        text_mask &= input_on_device != int(self.vision_start_token_id)
        text_mask &= input_on_device != int(self.vision_end_token_id)
        text_indices = text_mask.nonzero(as_tuple=True)[0]
        if text_indices.numel() == 0:
            return saliency

        query = F.normalize(hidden_norm[text_indices].mean(dim=0), dim=0, eps=1e-8)
        query_scores = (hidden_norm[visual_on_device] @ query).float()
        q_min = query_scores.min()
        q_span = (query_scores.max() - q_min).clamp_min(1e-6)
        query_scores = (query_scores - q_min) / q_span

        base_scores = saliency[visual_on_device].float()
        b_min = base_scores.min()
        b_span = (base_scores.max() - b_min).clamp_min(1e-6)
        base_scores = (base_scores - b_min) / b_span

        mixed = (1.0 - mix) * base_scores + mix * query_scores
        out = saliency.clone()
        out[visual_on_device] = mixed.to(dtype=out.dtype)
        return out

    def process_memory_streaming(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        input_ids: torch.Tensor,
        video_grid_thw: Optional[torch.Tensor] = None,
        pair_distance_threshold: Optional[float] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        if video_grid_thw is None:
            raise ValueError("video_grid_thw is None for EntityMem")

        batch_size, sequence_length, _ = hidden_states.shape
        device = hidden_states.device
        short_grids = max(1, int(self.short_term_frames / max(1, int(self.temporal_patch_size))))
        # Approximate old-frame token budget by the same frame count used by ObjectStream.
        old_token_budget_frames = max(1, int(self.mid_term_frames / max(1, int(self.temporal_patch_size))))

        processed_samples = []
        kept_indices_list: List[torch.Tensor] = []
        records = []

        for batch_index in range(batch_size):
            sample_input_ids = input_ids[batch_index]
            visual_indices = scan_visual_indices(sample_input_ids, self.vision_start_token_id, self.vision_end_token_id).cpu()
            if visual_indices.numel() == 0:
                kept = torch.arange(sequence_length, device=device)
                processed_samples.append(self._pack_sample(hidden_states, position_ids, position_embeddings, batch_index, kept))
                kept_indices_list.append(kept)
                continue

            time_ids = position_ids[0, batch_index]
            height_ids = position_ids[1, batch_index].to(torch.long)
            width_ids = position_ids[2, batch_index].to(torch.long)
            height_ids, width_ids = self._localize_spatial_ids(visual_indices, height_ids, width_ids)
            grid_hw = self._compute_grid_hw(visual_indices, height_ids, width_ids)
            frames, frame_to_indices, frame_to_grid = self._frame_maps(visual_indices, time_ids, height_ids, width_ids, grid_hw)

            keep_mask = torch.ones(sequence_length, dtype=torch.bool, device=device)
            keep_mask[visual_indices] = False
            hidden_norm = F.normalize(hidden_states[batch_index].float(), dim=1, eps=1e-8)

            # Query-agnostic saliency proxy: local novelty against each frame mean.
            saliency = torch.zeros(sequence_length, dtype=torch.float32, device=device)
            for frame_id in frames:
                idx = frame_to_indices[frame_id]
                feats = hidden_norm[idx]
                center = F.normalize(feats.mean(dim=0), dim=0, eps=1e-8)
                saliency[idx] = (1.0 - (feats @ center)).clamp_min(0.0)
                # Break ties on very static frames with embedding norm information.
                if float(saliency[idx].max().item()) <= 1e-8:
                    saliency[idx] = hidden_states[batch_index, idx].float().norm(dim=1)
            saliency = self._mix_query_saliency(saliency, hidden_norm, sample_input_ids, visual_indices)

            entity_states: dict[int, _EntityState] = {}
            entity_tokens: dict[int, list[int]] = {}
            entity_scores: dict[int, list[float]] = {}
            entity_times: dict[int, list[int]] = {}
            entity_chunks: dict[int, int] = {}
            entity_rank_stats: dict[int, tuple[int, float]] = {}

            old_frames = frames[:-short_grids] if len(frames) > short_grids else []
            recent_frames = set(frames[-short_grids:])

            chunk_enabled = bool(self.chunk_selection) and int(self.chunk_size_frames) > 0
            chunk_size = max(1, int(self.chunk_size_frames))
            active_chunk_id = None
            for frame_id in frames:
                idx = frame_to_indices[frame_id]
                if frame_id in recent_frames:
                    keep_mask[idx] = True
                    continue
                if chunk_enabled:
                    chunk_id = int(frame_id) // chunk_size
                    if active_chunk_id is None:
                        active_chunk_id = chunk_id
                    elif chunk_id != active_chunk_id:
                        entity_states = {}
                        active_chunk_id = chunk_id
                grid = frame_to_grid[frame_id]
                clusters = self._connected_components(idx, saliency, height_ids, width_ids, grid, device)
                for cluster in clusters:
                    cidx = torch.tensor(cluster, dtype=torch.long)
                    feature = self._weighted_mean(hidden_norm[cidx], saliency[cidx])
                    cluster_saliency = float(saliency[cidx].mean().item())
                    eid = self._match_or_create(entity_states, feature, cluster_saliency, frame_id)
                    entity_tokens.setdefault(eid, []).extend(cluster)
                    entity_scores.setdefault(eid, []).extend([float(x) for x in saliency[cidx].detach().cpu().tolist()])
                    entity_times.setdefault(eid, []).extend([int(frame_id)] * len(cluster))
                    if chunk_enabled:
                        entity_chunks.setdefault(eid, int(active_chunk_id if active_chunk_id is not None else 0))
                    st = entity_states.get(eid)
                    if st is not None:
                        entity_rank_stats[eid] = (int(st.persistence), float(st.saliency))
                self._evict_states(entity_states, frame_id)

            # Allocate per-entity token slots. More persistent/salient entities survive eviction first.
            old_budget = max(self.tokens_per_entity, old_token_budget_frames * max(1, grid_hw[0] * grid_hw[1] if grid_hw else self.tokens_per_entity) // 4)
            def _rank_key(eid: int):
                if eid in entity_rank_stats:
                    return entity_rank_stats[eid]
                st = entity_states.get(eid, _EntityState(eid, torch.empty(0), 0, 0.0))
                return int(st.persistence), float(st.saliency)

            if chunk_enabled:
                by_chunk: dict[int, list[int]] = {}
                for eid in entity_tokens.keys():
                    by_chunk.setdefault(int(entity_chunks.get(eid, 0)), []).append(eid)
                per_chunk_entities = int(self.chunk_max_entities) if int(self.chunk_max_entities) > 0 else int(self.max_entities)
                ranked_entities = []
                for chunk_id in sorted(by_chunk):
                    ranked_entities.extend(
                        sorted(by_chunk[chunk_id], key=_rank_key, reverse=True)[:per_chunk_entities]
                    )
                ranked_entities = sorted(set(ranked_entities), key=lambda eid: (int(entity_chunks.get(eid, 0)),) + tuple(_rank_key(eid)))
            else:
                ranked_entities = sorted(entity_tokens.keys(), key=_rank_key, reverse=True)[: self.max_entities]
            per_entity = max(1, min(self.tokens_per_entity, old_budget // max(1, len(ranked_entities))))
            kept_old = []
            for eid in ranked_entities:
                toks = entity_tokens.get(eid, [])
                if not toks:
                    continue
                tidx = torch.tensor(toks, dtype=torch.long)
                scores = saliency[tidx]
                k = min(per_entity, int(tidx.numel()))
                if self.temporal_selection:
                    frame_ids = entity_times.get(eid, [int(time_ids[int(x)].item()) for x in tidx.detach().cpu().tolist()])
                    chosen = self._select_temporal_tokens(
                        tidx,
                        scores,
                        frame_ids,
                        k,
                        boundary_first=self.temporal_boundary_selection,
                        even_coverage=self.temporal_even_selection,
                        novelty_coverage=self.temporal_novelty_selection,
                    )
                else:
                    chosen = tidx[scores.detach().cpu().topk(k).indices].sort().values
                keep_mask[chosen] = True
                kept_old.extend([int(x) for x in chosen.detach().cpu().tolist()])

            anchor_tokens = self._select_temporal_anchor_tokens(old_frames, frame_to_indices, saliency, keep_mask)
            if anchor_tokens.numel() > 0:
                anchor_tokens = anchor_tokens.to(device=keep_mask.device)
                keep_mask[anchor_tokens] = True
                kept_old.extend([int(x) for x in anchor_tokens.detach().cpu().tolist()])

            keyframe_tokens = self._select_temporal_keyframe_tokens(
                old_frames,
                frame_to_indices,
                saliency,
                keep_mask,
                height_ids=height_ids,
                width_ids=width_ids,
            )
            if keyframe_tokens.numel() > 0:
                keyframe_tokens = keyframe_tokens.to(device=keep_mask.device)
                keep_mask[keyframe_tokens] = True
                kept_old.extend([int(x) for x in keyframe_tokens.detach().cpu().tolist()])

            event_tokens = self._select_temporal_event_tokens(
                old_frames,
                frame_to_indices,
                saliency,
                hidden_norm,
                keep_mask,
            )
            if event_tokens.numel() > 0:
                event_tokens = event_tokens.to(device=keep_mask.device)
                keep_mask[event_tokens] = True
                kept_old.extend([int(x) for x in event_tokens.detach().cpu().tolist()])

            burst_tokens = self._select_temporal_burst_tokens(
                entity_tokens,
                entity_times,
                saliency,
                hidden_norm,
                keep_mask,
            )
            if burst_tokens.numel() > 0:
                burst_tokens = burst_tokens.to(device=keep_mask.device)
                keep_mask[burst_tokens] = True
                kept_old.extend([int(x) for x in burst_tokens.detach().cpu().tolist()])

            recap_tokens = self._select_temporal_recap_tokens(
                old_frames,
                frame_to_indices,
                saliency,
                hidden_norm,
                keep_mask,
            )
            if recap_tokens.numel() > 0:
                recap_tokens = recap_tokens.to(device=keep_mask.device)
                keep_mask[recap_tokens] = True
                kept_old.extend([int(x) for x in recap_tokens.detach().cpu().tolist()])

            kept = keep_mask.nonzero(as_tuple=True)[0]
            sample_hidden = hidden_states[batch_index]
            if self.temporal_content_encoding or self.temporal_order_content_encoding:
                old_kept = torch.tensor(sorted(set(kept_old)), dtype=torch.long, device=keep_mask.device)
                sample_hidden = self._apply_temporal_content_encoding(sample_hidden, old_kept, time_ids)
                hidden_for_pack = hidden_states.clone()
                hidden_for_pack[batch_index] = sample_hidden
            else:
                hidden_for_pack = hidden_states
            processed_samples.append(self._pack_sample(hidden_for_pack, position_ids, position_embeddings, batch_index, kept))
            kept_indices_list.append(kept)

            records.append({
                "batch_idx": int(batch_index),
                "method": "entitymem",
                "frames": len(frames),
                "old_frames": len(old_frames),
                "recent_frames": len(recent_frames),
                "entities": len(entity_states),
                "kept_tokens": int(kept.numel()),
                "visual_tokens": int(visual_indices.numel()),
                "kept_visual_tokens": int(keep_mask[visual_indices].sum().item()),
                "temporal_entity_selection": bool(self.temporal_selection),
                "temporal_boundary_selection": bool(self.temporal_boundary_selection),
                "temporal_even_selection": bool(self.temporal_even_selection),
                "temporal_novelty_selection": bool(self.temporal_novelty_selection),
                "temporal_gap_reset": bool(self.temporal_gap_reset),
                "temporal_gap_reset_frames": None if self.temporal_gap_reset_frames is None else int(self.temporal_gap_reset_frames),
                "temporal_anchor_selection": bool(self.temporal_anchor_selection),
                "temporal_anchor_bins": int(self.temporal_anchor_bins),
                "temporal_anchor_tokens_per_bin": int(self.temporal_anchor_tokens_per_bin),
                "temporal_keyframe_selection": bool(self.temporal_keyframe_selection),
                "temporal_keyframe_max_frames": int(self.temporal_keyframe_max_frames),
                "temporal_keyframe_tokens_per_frame": int(self.temporal_keyframe_tokens_per_frame),
                "temporal_keyframe_grid_selection": bool(self.temporal_keyframe_grid_selection),
                "temporal_keyframe_grid_size": int(self.temporal_keyframe_grid_size),
                "temporal_event_selection": bool(self.temporal_event_selection),
                "temporal_event_max_frames": int(self.temporal_event_max_frames),
                "temporal_event_tokens_per_frame": int(self.temporal_event_tokens_per_frame),
                "temporal_burst_selection": bool(self.temporal_burst_selection),
                "temporal_burst_max_events": int(self.temporal_burst_max_events),
                "temporal_burst_max_events_per_entity": int(getattr(self, "temporal_burst_max_events_per_entity", 0)),
                "temporal_burst_radius": int(self.temporal_burst_radius),
                "temporal_burst_tokens_per_frame": int(self.temporal_burst_tokens_per_frame),
                "chunk_selection": bool(self.chunk_selection),
                "chunk_size_frames": int(self.chunk_size_frames),
                "chunk_max_entities": int(self.chunk_max_entities),
                "temporal_recap_selection": bool(self.temporal_recap_selection),
                "temporal_recap_bins": int(self.temporal_recap_bins),
                "temporal_recap_tokens_per_bin": int(self.temporal_recap_tokens_per_bin),
                "temporal_content_encoding": bool(self.temporal_content_encoding),
                "temporal_content_scale": float(self.temporal_content_scale),
                "temporal_order_content_encoding": bool(self.temporal_order_content_encoding),
                "temporal_order_content_scale": float(self.temporal_order_content_scale),
                "query_saliency": bool(self.query_saliency),
                "query_saliency_mix": float(self.query_saliency_mix),
                "short_frames": int(self.short_term_frames),
                "medium_frames": int(self.mid_term_frames),
                "tokens_per_entity": int(self.tokens_per_entity),
                "max_entities": int(self.max_entities),
            })

        if self.save_path:
            with open(self.save_path, "a") as f:
                for rec in records:
                    f.write(json.dumps(rec) + "\n")

        hidden_out, pos_emb_out, pos_ids_out, attn_out = right_pad_and_stack(
            processed_samples, pos_key1="pos_emb1", pos_key2="pos_emb2"
        )
        return hidden_out, pos_emb_out, pos_ids_out, attn_out, kept_indices_list
