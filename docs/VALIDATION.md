# Release validation, 2026-10-15

Tests use isolated pinned dependency overlays and `CUDA_VISIBLE_DEVICES=''` on
Linux; shared runtime installations and ongoing GPU jobs are not modified.

| Check | Result |
|---|---|
| Transformers4.50.0 + Accelerate1.4.0 + NumPy1.26.4, Qwen2.5 model/processor/utilities and all issue#1 legacy imports | Passed |
| Qwen2.5 test selection | 6passed,1 Qwen3.5-only test skipped |
| Transformers5.18.0 native Qwen3.5 and shipped adapter imports | Passed |
| Qwen3.5 test selection | 6passed,1 Qwen2.5-only test skipped |
| Actual production EntityMem tensor contract / recent8 retention | Passed |
| Both CLI dry-run / label-independent manifest construction | Passed |
| Real-video native Qwen3.5 processor input IDs and positional layout | Exact equality,900 visual tokens, no model weights/GPU loaded |
| Qwen3.5 exported core versus validated source | EntityMem/utils byte-identical; adapter AST equal except relative imports, local-files parameter and sharded-device transfer |
| Three installable distributions | Wheel build and contents checked |

CPU validation ran on Python3.12. The Qwen2.5 installation recipe continues
to recommend Python3.11 and its pinned4.50 environment; Qwen3.5 uses a separate
Python3.12/5.18 environment. This document does not claim a fresh full73.3
benchmark reproduction or a new Qwen3.5 GPU benchmark run. Existing real-model
native-logit and memory-cache checks belong to the validated source experiments,
not to these CPU-only packaging tests.

Run the matching test suite after installation:

```bash
OBJECTSTREAM_TEST_BACKBONE=qwen25 python -m pytest tests -q
# In the separate Qwen3.5 environment:
OBJECTSTREAM_TEST_BACKBONE=qwen35 python -m pytest tests -q
```

Optional actual-video input parity check (local model config/processor assets,
not model weights):

```bash
CUDA_VISIBLE_DEVICES='' python scripts/validate_qwen35_inputs.py \
  --model_path /path/to/Qwen3.5-9B --video_path /path/to/video.mp4 --timestamp 10
```
