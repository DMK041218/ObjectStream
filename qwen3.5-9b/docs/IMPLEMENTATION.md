# Implementation Notes

## Native model integration

The adapter uses the native Transformers Qwen3.5 visual encoder, processor,
and hybrid decoder without changing their model parameters. Retained visual
features are packed with their source timestamps and native temporal / spatial
positions. The implementation is in `objectstream/backends/qwen35/`.

## Memory across queries

A runner instance stores compressed visual features, source-time geometry,
and the last observed video time. Queries must be chronological; equal-time
queries reuse the prepared visual memory, while earlier timestamps are
rejected. `reset()` clears the cache between independent episodes.

This is visual-feature memory, not an LLM KV cache or dialogue-history cache.
Entity states are reconstructed during compression of the retained features;
entity IDs are not persisted across separate calls.

The default reader samples at 1 fps, anchored at the query timestamp, with at
most 256 new sampled frames per update. If updates are widely separated,
that cap can leave intervening footage unobserved. The budget parameters are
upper bounds rather than a fixed percentage of retained tokens.

## Complete-video segment ingestion

For full-video evaluation or an external frame source, repeatedly call:

```python
runner.ingest_frames(video_id, frames, source_indices, source_pts, source_fps)
```

`frames` is a raw RGB tensor in TCHW order with an even number of frames,
at least four. Source frame indices and presentation timestamps must align
with the frames and be chronological. Any padding must duplicate already
observed frames, never future frames. Ingestion updates memory without
generating an answer. After the last segment, call `answer()` with the same
video ID and its final observed timestamp.

Use one runner per process: video-sampling instrumentation is not thread-safe.
When sharding evaluation, keep all questions from a video in the same worker.

## Attribution

Video reading includes utilities derived from Qwen and FluxMem. Original
copyright headers are retained. The native model and processor are provided
by the separately installed Transformers package. Model weights and benchmark
videos are not redistributed. See [NOTICE](../../NOTICE) and [LICENSE](../LICENSE).
