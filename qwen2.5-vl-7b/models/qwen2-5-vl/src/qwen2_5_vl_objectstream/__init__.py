# Copyright 2025 The Qwen Team and The HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0.
from typing import TYPE_CHECKING

from transformers.utils import _LazyModule
from transformers.utils.import_utils import define_import_structure


if TYPE_CHECKING:
    from .configuration_qwen2_5_vl_objectstream import *
    from .modeling_qwen2_5_vl_objectstream import *
    from .processing_qwen2_5_vl_objectstream import *
    from .objectstream import *
else:
    import sys

    _file = globals()["__file__"]
    sys.modules[__name__] = _LazyModule(
        __name__,
        _file,
        define_import_structure(_file),
        module_spec=__spec__,
    )
