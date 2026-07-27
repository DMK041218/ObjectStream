"""Optional heavy entity frontends used only by the OVO ablation."""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, Dinov2Model, SamModel, SamProcessor


@dataclass
class ExternalRegions:
    regions: list[list[tuple[list[int], torch.Tensor]]]
    latency_s: float
    fallback: bool


class ExternalEntityFrontend:
    def __init__(self) -> None:
        self.kind = "spatial_cluster"
        self.device: torch.device | None = None
        self.dino_processor = None
        self.dino = None
        self.sam_processor = None
        self.sam = None

    def configure(self, kind: str, device: torch.device) -> None:
        self.kind = kind
        self.device = device
        if kind in {"dinov2_cluster", "sam_dinov2"} and self.dino is None:
            self.dino_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base", local_files_only=True)
            self.dino = Dinov2Model.from_pretrained(
                "facebook/dinov2-base", torch_dtype=torch.float16, local_files_only=True
            ).to(device).eval()
        if kind == "sam_dinov2" and self.sam is None:
            self.sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base", local_files_only=True)
            self.sam = SamModel.from_pretrained(
                "facebook/sam-vit-base", torch_dtype=torch.float16, local_files_only=True
            ).to(device).eval()

    @staticmethod
    def _sample_frames(path: str, count: int) -> list[np.ndarray]:
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            return []
        indices = np.linspace(0, total - 1, max(1, count), dtype=np.int64)
        frames = []
        for frame_idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ok, frame = cap.read()
            if ok:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return frames

    @torch.inference_mode()
    def _dino_maps(self, frames: list[np.ndarray], out_hw: tuple[int, int]) -> torch.Tensor:
        maps = []
        for start in range(0, len(frames), 8):
            batch = self.dino_processor(images=frames[start : start + 8], return_tensors="pt")
            pixel_values = batch["pixel_values"].to(self.device, dtype=torch.float16)
            hidden = self.dino(pixel_values=pixel_values).last_hidden_state[:, 1:]
            side = int(round(hidden.shape[1] ** 0.5))
            hidden = hidden.transpose(1, 2).reshape(hidden.shape[0], hidden.shape[2], side, side)
            hidden = F.interpolate(hidden.float(), size=out_hw, mode="bilinear", align_corners=False)
            maps.append(F.normalize(hidden, dim=1).cpu())
        return torch.cat(maps, dim=0)

    @staticmethod
    def _feature_components(feature_map: torch.Tensor, threshold: float) -> list[list[int]]:
        channels, height, width = feature_map.shape
        flat = feature_map.reshape(channels, -1).transpose(0, 1)
        seen = set()
        components = []
        for start in range(height * width):
            if start in seen:
                continue
            seen.add(start)
            stack = [start]
            group = []
            while stack:
                cur = stack.pop()
                group.append(cur)
                y, x = divmod(cur, width)
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    nxt = ny * width + nx
                    if (
                        0 <= ny < height
                        and 0 <= nx < width
                        and nxt not in seen
                        and float(flat[cur] @ flat[nxt]) >= threshold
                    ):
                        seen.add(nxt)
                        stack.append(nxt)
            if len(group) >= 2:
                components.append(group)
        return components

    @torch.inference_mode()
    def _sam_masks(self, frame: np.ndarray, out_hw: tuple[int, int]) -> list[list[int]]:
        height, width = frame.shape[:2]
        points = [
            [float(x), float(y)]
            for y in np.linspace(height * 0.1, height * 0.9, 4)
            for x in np.linspace(width * 0.1, width * 0.9, 4)
        ]
        inputs = self.sam_processor(
            images=frame,
            input_points=[[points]],
            return_tensors="pt",
        )
        original_sizes = inputs.pop("original_sizes")
        reshaped_sizes = inputs.pop("reshaped_input_sizes")
        inputs = {
            k: v.to(self.device, dtype=torch.float16) if torch.is_floating_point(v) else v.to(self.device)
            for k, v in inputs.items()
        }
        outputs = self.sam(**inputs, multimask_output=False)
        masks = self.sam_processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(), original_sizes, reshaped_sizes
        )[0]
        if masks.ndim == 4:
            masks = masks[0]
        masks = F.interpolate(
            masks[:, None].float(), size=out_hw, mode="nearest"
        )[:, 0].bool()
        regions = []
        occupied = torch.zeros(out_hw, dtype=torch.bool)
        for mask in masks:
            mask = mask & ~occupied
            idx = mask.flatten().nonzero(as_tuple=True)[0].tolist()
            if len(idx) >= 2:
                regions.append(idx)
                occupied |= mask
        remainder = (~occupied).flatten().nonzero(as_tuple=True)[0].tolist()
        if len(remainder) >= 2:
            regions.append(remainder)
        return regions

    def build(
        self,
        kind: str,
        video_path: str,
        frame_count: int,
        grid_hw: tuple[int, int],
        match_threshold: float,
    ) -> ExternalRegions:
        started = time.perf_counter()
        frames = self._sample_frames(video_path, frame_count)
        if len(frames) != frame_count:
            return ExternalRegions([], time.perf_counter() - started, True)
        dino_maps = self._dino_maps(frames, grid_hw)
        result = []
        sam_anchors = {}
        if kind == "sam_dinov2":
            anchor_ids = sorted(set(range(0, len(frames), 8)) | {len(frames) - 1})
            for anchor_id in anchor_ids:
                sam_anchors[anchor_id] = self._sam_masks(frames[anchor_id], grid_hw)
        for frame_idx, frame in enumerate(frames):
            if kind == "dinov2_cluster":
                groups = self._feature_components(dino_maps[frame_idx], match_threshold)
            else:
                anchor_id = min(sam_anchors, key=lambda idx: abs(idx - frame_idx))
                anchor_features = dino_maps[anchor_id].flatten(1).transpose(0, 1)
                prototypes = []
                for group in sam_anchors[anchor_id]:
                    idx = torch.tensor(group, dtype=torch.long)
                    prototypes.append(F.normalize(anchor_features[idx].mean(dim=0), dim=0, eps=1e-8))
                current = dino_maps[frame_idx].flatten(1).transpose(0, 1)
                if prototypes:
                    assignment = current @ torch.stack(prototypes).transpose(0, 1)
                    labels = assignment.argmax(dim=1)
                    groups = [
                        (labels == region_id).nonzero(as_tuple=True)[0].tolist()
                        for region_id in range(len(prototypes))
                    ]
                    groups = [group for group in groups if len(group) >= 2]
                else:
                    groups = self._feature_components(dino_maps[frame_idx], match_threshold)
            flat_features = dino_maps[frame_idx].flatten(1).transpose(0, 1)
            regions = []
            for group in groups:
                idx = torch.tensor(group, dtype=torch.long)
                feature = F.normalize(flat_features[idx].mean(dim=0), dim=0, eps=1e-8)
                regions.append((group, feature))
            result.append(regions)
        return ExternalRegions(result, time.perf_counter() - started, not any(result))
