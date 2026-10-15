"""Native Qwen3.5 imports without any Qwen2.5 packages."""


def test_native_model_and_runner_imports():
    from objectstream import Qwen35ObjectStream
    from objectstream.backends.qwen35.backend import Backend, pack_layout
    from transformers import Qwen3_5ForConditionalGeneration

    assert Qwen35ObjectStream and Backend and Qwen3_5ForConditionalGeneration
    assert callable(pack_layout)
