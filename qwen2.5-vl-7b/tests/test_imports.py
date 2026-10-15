"""Model, processor, and evaluation imports in the Qwen2.5 environment."""
import importlib


def test_model_processor_and_legacy_imports():
    from qwen_vl_utils_objectstream import process_vision_info
    from qwen2_5_vl_objectstream import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
    for name in ('objectstream','configuration_qwen2_5_vl_objectstream','utils',
                 'processing_qwen2_5_vl_objectstream','modeling_qwen2_5_vl_objectstream','entitymem'):
        importlib.import_module('ovo_ablation_work.models.'+name)
    assert callable(process_vision_info)
    assert Qwen2_5_VLProcessor and Qwen2_5_VLForConditionalGeneration
