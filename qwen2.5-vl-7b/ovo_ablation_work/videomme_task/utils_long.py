from loguru import logger as eval_logger

from lmms_eval.tasks.videomme.utils import (
    videomme_aggregate_results,
    videomme_doc_to_text,
    videomme_doc_to_visual,
    videomme_process_results,
)


def videomme_long_process_docs(dataset):
    filtered = dataset.filter(lambda doc: doc.get("duration") == "long")
    eval_logger.info(f"VideoMME long filter: {len(filtered)} / {len(dataset)} docs")
    return filtered
