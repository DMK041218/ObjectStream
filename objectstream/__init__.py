"""Entity-centric ObjectStream helpers for Qwen2.5-VL streaming scripts.

Heavy ML dependencies are imported lazily by submodules so command-line dry
runs work before installing Torch, Transformers, or the ObjectStream packages.
"""

from .config import EntityMemoryConfig, QwenObjectStreamConfig

__all__ = ["EntityMemoryConfig", "QwenObjectStreamConfig"]
