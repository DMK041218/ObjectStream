# Missing modules in the old export (issue#1)

The issue references commit `4ad60cc`. Do not attempt to repair that historical
snapshot by renaming the public Transformers processor or public video utils:
that can change preprocessing and memory integration.

Update to the latest main branch and run the Qwen2.5 installer:

```bash
git pull --ff-only
PYTHON_BIN=python3.11 VENV_DIR=.venv-qwen25 bash scripts/install_environment.sh
source .venv-qwen25/bin/activate
python scripts/check_environment.py --backbone qwen25
OBJECTSTREAM_TEST_BACKBONE=qwen25 python -m pytest tests -q
```

| Missing import | Shipped source |
|---|---|
| `qwen_vl_utils_objectstream` | `models/qwen-vl-utils/src/qwen_vl_utils_objectstream/` |
| `qwen2_5_vl_objectstream.Qwen2_5_VLProcessor` | `models/qwen2-5-vl/src/qwen2_5_vl_objectstream/processing_qwen2_5_vl_objectstream.py` |
| `.objectstream` | canonical module above, plus legacy shim in `ovo_ablation_work/models/` |
| `.configuration_qwen2_5_vl_objectstream` | canonical config, plus legacy shim |
| `.utils` | canonical scan/padding helpers, plus legacy shim |

The regression test imports the actual model class, processor, both utility
packages and every listed legacy relative import. Installation metadata uses
explicit wheel package paths. StreamingBench now fails visibly if the patched
packages are unavailable instead of falling back to a different utility.

This is an import/packaging fix. It is not a claim that a new full 73.3-score
reproduction was performed. The original archived e64/t8/burst profile is
documented in [Evaluation](EVALUATION.md); minimal quick-start defaults are not
the same as that profile. Retain sampling, pixel budget, task subset and scorer
when comparing results. Qwen3.5 scores are a separate-backbone experiment.
