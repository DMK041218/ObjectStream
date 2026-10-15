#!/usr/bin/env python3
from __future__ import annotations

import importlib
import platform
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def require(module_name: str) -> object:
    try:
        return importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - diagnostic entry point
        raise SystemExit(f"[FAIL] cannot import {module_name}: {exc}") from exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', choices=('qwen25', 'qwen35'), default='qwen25')
    args = parser.parse_args()
    expected_python = (3, 11) if args.backbone == 'qwen25' else (3, 12)
    if sys.version_info[:2] != expected_python:
        raise SystemExit(
            f"[FAIL] expected Python {expected_python}, found {platform.python_version()}"
        )

    torch = require("torch")
    transformers = require("transformers")
    require("objectstream")
    if args.backbone == 'qwen25':
        require("qwen_vl_utils_objectstream")
        backend = require("qwen2_5_vl_objectstream")
        for symbol in ("Qwen2_5_VLForConditionalGeneration", "Qwen2_5_VLProcessor"):
            if not hasattr(backend, symbol):
                raise SystemExit(f"[FAIL] qwen2_5_vl_objectstream lacks {symbol}")
        for module in ('objectstream', 'configuration_qwen2_5_vl_objectstream', 'utils', 'processing_qwen2_5_vl_objectstream'):
            require('ovo_ablation_work.models.' + module)
        require('ovo_ablation_work.models.modeling_qwen2_5_vl_objectstream')
        require('ovo_ablation_work.models.entitymem')
        expected = '4.50.0'
    else:
        require('objectstream.backends.qwen35.backend')
        if not hasattr(transformers, 'Qwen3_5ForConditionalGeneration'):
            raise SystemExit('[FAIL] missing native Qwen3.5 implementation')
        expected = '5.18.0'
    if transformers.__version__ != expected:
        raise SystemExit(f'[FAIL] {args.backbone} requires transformers=={expected}; use separate environments')

    print(f"[OK] Python: {platform.python_version()}")
    print(f"[OK] PyTorch: {torch.__version__}")
    print(f"[OK] Transformers: {transformers.__version__}")
    print(f"[OK] CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[OK] CUDA runtime: {torch.version.cuda}")
        print(f"[OK] GPU: {torch.cuda.get_device_name(0)}")


if __name__ == "__main__":
    main()
