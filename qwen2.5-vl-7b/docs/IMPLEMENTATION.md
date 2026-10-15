# Implementation Notes

## Model and preprocessing

The modified model, configuration, processor, and memory modules are packaged
under `models/qwen2-5-vl/src/qwen2_5_vl_objectstream/`. Video preprocessing is
packaged under `models/qwen-vl-utils/src/qwen_vl_utils_objectstream/`.
Install both local packages with the pinned Transformers 4.50.0 environment.

The public `use_objectstream=True` argument selects the entity-centric
`EntityMem` path in the modified model. The legacy
`ovo_ablation_work/models/` import paths resolve their supporting modules
through the installed packages. `objectstream/entity_memory.py` is a
standalone helper rather than the memory implementation used by the model.

## Temporal memory

The model processes the supplied video prefix chronologically and constructs
object memory across its frames. Each independent generation call initializes
this memory again; it does not inherit a visual-memory cache from previous
queries. To evaluate a later query, supply the corresponding observed prefix.

## Attribution

The modified model and processor derive from Qwen2.5-VL and Hugging Face
Transformers. The bundled video utilities and legacy memory helper include
code derived from Qwen and FluxMem. Original copyright headers are retained;
see the repository's [NOTICE](../../NOTICE) and [LICENSE](../LICENSE).
