"""Record actual source samples; identical sampler instrumentation to validated runs."""
from contextlib import contextmanager


@contextmanager
def capture_decord_samples():
    """Observe actual get_batch indices/PTS without changing reader selection.

    This instrumentation temporarily wraps Decord's reader factory. Use one
    model worker per process, not concurrent threads within a process.
    """
    import decord
    original = decord.VideoReader
    traces = []

    class Reader:
        def __init__(self, *args, **kwargs):
            self.reader = original(*args, **kwargs)

        def __len__(self):
            return len(self.reader)

        def __getattr__(self, name):
            return getattr(self.reader, name)

        def get_batch(self, indices):
            indices = [int(x) for x in indices]
            result = self.reader.get_batch(indices)
            pts = self.reader.get_frame_timestamp(indices)
            if hasattr(pts, 'asnumpy'):
                pts = pts.asnumpy()
            traces.append({'reader': 'decord', 'source_fps': float(self.reader.get_avg_fps()),
                           'indices': indices, 'pts': [float(x[0]) for x in pts]})
            return result

    decord.VideoReader = Reader
    try:
        yield traces
    finally:
        decord.VideoReader = original
