"""Portable Qwen3.5 runner; imports Transformers only when loading a model."""
from pathlib import Path
import math


class Qwen35ObjectStream:
    def __init__(self, model_path='Qwen/Qwen3.5-9B', cards=1, local_files_only=False,
                 mode='objectstream'):
        if mode not in ('objectstream', 'native'):
            raise ValueError('mode must be objectstream or native')
        import torch
        if torch.cuda.device_count() != cards:
            raise ValueError('Set CUDA_VISIBLE_DEVICES to exactly the requested cards before loading')
        from .backends.qwen35.backend import Backend
        self.backend = Backend(model_path, cards=cards,
                               local_files_only=local_files_only)
        self.mode = mode

    def reset(self):
        """Reset visual memory between unrelated videos or evaluation episodes."""
        self.backend.cache.clear()

    def answer(self, video_path, question, timestamp):
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError('Timestamp must be a finite nonnegative source-video time')
        response, telemetry, _ = self.backend.answer(Path(video_path), question,
                                                    float(timestamp), self.mode)
        return {'answer': response, 'telemetry': telemetry}

    def ingest_frames(self, video_id, frames, source_indices, source_pts, source_fps):
        """Question-free segment ingestion, then answer() at the final observed PTS.

        frames:TCHW raw RGB tensor, >=4 and even length; source indices/PTS
        correspond to the actual decoded frames, including any past-only pad.
        This explicitly supports frequent updates without generating an answer.
        """
        if self.mode != 'objectstream':
            raise ValueError('ingest_frames is a compressed-memory operation')
        if len(frames) < 4 or len(frames) % 2:
            raise ValueError('Provide an even number of frames >=4; pad with observed frames only')
        if len(source_indices) != len(frames) or len(source_pts) != len(frames):
            raise ValueError('Frame/trace lengths differ')
        if list(source_indices) != sorted(source_indices) or list(source_pts) != sorted(source_pts):
            raise ValueError('Source indices/PTS must be chronological')
        if source_fps <= 0 or not all(math.isfinite(p) and p >= 0 for p in source_pts):
            raise ValueError('Invalid source frame metadata')
        trace = {'indices': list(source_indices), 'pts': list(source_pts),
                 'source_fps': source_fps, 'reader': 'provided_observed_frames'}
        self.backend._provided_segment = (frames, trace)
        self.backend._ingest_only = True
        try:
            _, telemetry, _ = self.backend.answer(Path(video_id), 'Observe the video.',
                                                  float(source_pts[-1]), 'objectstream')
        finally:
            self.backend._provided_segment = None
            self.backend._ingest_only = False
        return telemetry
