"""Optional LMMS-Eval task loading; no video or dataset download is needed."""
from pathlib import Path

import pytest

pytest.importorskip("lmms_eval", reason="Install requirements/videomme.txt for task support")
from datasets import Dataset
from lmms_eval.utils import load_yaml_config


def test_videomme_long_task_functions_load():
    root = Path(__file__).resolve().parents[1]
    config = load_yaml_config(str(root / "scripts/videomme_long.yaml"))
    for name in ("process_docs", "doc_to_visual", "doc_to_text", "process_results"):
        assert callable(config[name])
    assert callable(config["metric_list"][0]["aggregation"])
    dataset = Dataset.from_list([
        {"duration": "short", "question_id": "short0"},
        {"duration": "long", "question_id": "long0"},
        {"duration": "medium", "question_id": "medium0"},
    ])
    selected = config["process_docs"](dataset)
    assert selected["question_id"] == ["long0"]
    prompt = config["doc_to_text"](
        {"question": "Which object is visible?", "options": ["A. Cup", "B. Book", "C. Pen", "D. Ball"]},
        {"post_prompt": "Answer with the option's letter only."},
    )
    assert "Which object is visible?" in prompt and "A. Cup" in prompt
