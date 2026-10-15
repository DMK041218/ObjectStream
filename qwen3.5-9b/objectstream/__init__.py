"""Training-free ObjectStream integration for the native Qwen3.5 model.

The public runner loads heavy model dependencies only when instantiated.
"""

from .qwen35_runner import Qwen35ObjectStream

__all__ = ["Qwen35ObjectStream"]
