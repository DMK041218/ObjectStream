# Source and reproduction boundaries

## Qwen2.5-VL-7B

The complete modified model/config/processor/memory/utilities are under
`models/qwen2-5-vl/src/qwen2_5_vl_objectstream/`. The pinned dependency is
Transformers4.50.0; this is NOT an instruction to replace the bundled processor
with whichever public processor is currently installed. Video utilities are
the bundled `qwen_vl_utils_objectstream`, adapted from Qwen/FluxMem utilities.
Original copyright headers and Apache2.0 LICENSE are retained.

`use_objectstream=True` selects the entity-centric EntityMem path in the
installable model. The `.objectstream.ObjectStream` helper is retained for
legacy import compatibility; it is not the standalone `EntityMemoryBank`
demonstrator in `objectstream/entity_memory.py`. That demonstrator's tests
alone are insufficient to validate the production patched model imports.

The legacy `ovo_ablation_work/models` relative dependencies now resolve through
small shims to installed canonical files. Main evaluation entry points import
the installed model package, not a second copy of the model by file path.
Qwen2.5's production path builds temporal memory inside each supplied prefill;
separate generate calls do NOT automatically inherit a persistent visual bank.

## Qwen3.5-9B

`objectstream/backends/qwen35/backend.py` is the validated full-profile native
adapter, including the later provided-segment/ingest-only hooks and safe model
splitting. Relative imports and a `local_files_only` parameter replace private
machine paths. Model weights and the Transformers decoder/processor are not
vendored or modified. EntityMem/utils are byte-identical to the validated
default core. Sampling observation preserves actual Decord indices and PTS;
the bundled `_video.py` is the same reader utility, not a2.5 resize applied to
Qwen3.5 input. Native preprocessing uses factor32 and total-clip pixel budgets.

New packaging tests cover imports, actual production-memory tensor contracts,
dependency-light dry runs, label-independent prompts and sparse layout. Source
checks compare the exported adapter/core against validated source snapshots.
Fresh full model benchmark runs are NOT implied by a packaging update; no
ongoing experiment is interrupted to manufacture a new release validation.

These two backbone ports are not numerically interchangeable. Also distinguish
streaming query-time input windows from full-video offline ingestion. Check
sampling, source coverage, generation, scoring, actual retained tokens and
timing scope before comparing scores. In particular, the17.5% experimental
allocator and the adapted FluxMem baseline are not included in this release.
