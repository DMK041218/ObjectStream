#!/usr/bin/env python3
"""Check the independent native Qwen3.5 environment without loading weights."""
import importlib
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def require(module_name):
    try:
        return importlib.import_module(module_name)
    except Exception as exc:
        raise SystemExit(f"[FAIL] cannot import {module_name}: {exc}") from exc


def main():
    if sys.version_info[:2] != (3, 12):
        raise SystemExit(f"[FAIL] expected Python 3.12, found {platform.python_version()}")
    torch = require("torch")
    transformers = require("transformers")
    if transformers.__version__ != "5.18.0":
        raise SystemExit("[FAIL] Qwen3.5 requires transformers==5.18.0; use its separate environment")
    require("objectstream.qwen35_runner")
    require("objectstream.backends.qwen35.backend")
    if not hasattr(transformers, "Qwen3_5ForConditionalGeneration"):
        raise SystemExit("[FAIL] missing native Qwen3.5 implementation")
    print(f"[OK] Python: {platform.python_version()}")
    print(f"[OK] PyTorch: {torch.__version__}")
    print(f"[OK] Transformers: {transformers.__version__}")
    print(f"[OK] CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[OK] CUDA runtime: {torch.version.cuda}")
        print(f"[OK] GPU: {torch.cuda.get_device_name(0)}")


if __name__ == "__main__":
    main()
