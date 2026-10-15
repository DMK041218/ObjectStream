"""Regressions for issue#1; use the matching per-backbone environment."""
import importlib
import os
import pytest


@pytest.mark.skipif(os.environ.get('OBJECTSTREAM_TEST_BACKBONE') != 'qwen25', reason='Qwen2.5 separate environment')
def test_all_issue1_imports():
    from qwen_vl_utils_objectstream import process_vision_info
    from qwen2_5_vl_objectstream import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
    for name in ('objectstream','configuration_qwen2_5_vl_objectstream','utils',
                 'processing_qwen2_5_vl_objectstream','modeling_qwen2_5_vl_objectstream','entitymem'):
        importlib.import_module('ovo_ablation_work.models.'+name)
    assert callable(process_vision_info)
    assert Qwen2_5_VLProcessor and Qwen2_5_VLForConditionalGeneration


@pytest.mark.skipif(os.environ.get('OBJECTSTREAM_TEST_BACKBONE') != 'qwen35', reason='Qwen3.5 separate environment')
def test_qwen35_import():
    from objectstream.backends.qwen35.backend import Backend, pack_layout
    from transformers import Qwen3_5ForConditionalGeneration
    assert Backend and callable(pack_layout) and Qwen3_5ForConditionalGeneration
