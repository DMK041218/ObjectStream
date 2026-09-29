#!/usr/bin/env python3
from __future__ import annotations

import importlib
import platform
import sys


def require(module_name: str) -> object:
    try:
        return importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - diagnostic entry point
        raise SystemExit(f"[FAIL] cannot import {module_name}: {exc}") from exc


def main() -> None:
    if sys.version_info[:2] != (3, 11):
        raise SystemExit(
            f"[FAIL] expected Python 3.11, found {platform.python_version()}"
        )

    torch = require("torch")
    transformers = require("transformers")
    require("qwen_vl_utils_objectstream")
    backend = require("qwen2_5_vl_objectstream")
    require("objectstream")

    for symbol in (
        "Qwen2_5_VLForConditionalGeneration",
        "Qwen2_5_VLProcessor",
    ):
        if not hasattr(backend, symbol):
            raise SystemExit(f"[FAIL] qwen2_5_vl_objectstream lacks {symbol}")

    print(f"[OK] Python: {platform.python_version()}")
    print(f"[OK] PyTorch: {torch.__version__}")
    print(f"[OK] Transformers: {transformers.__version__}")
    print(f"[OK] CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[OK] CUDA runtime: {torch.version.cuda}")
        print(f"[OK] GPU: {torch.cuda.get_device_name(0)}")


if __name__ == "__main__":
    main()
