"""Validate and post-process Open Targets Platform evidence.

Polars entry point for the `evidence_postprocess_*` config.yaml steps, replacing
`pts.pyspark.evidence_postprocess`. It turns otter's `source`/`destination`/`settings` into the
parameters the recipe needs, reads, runs `pts.transformers.evidence.postprocess`, and writes. The
processing itself lives in the `pts.transformers.evidence` package.

It stays here rather than inside that package because otter resolves a transformer by importing
`pts.transformers.<name>` and taking the attribute of the same name, so a step module cannot live
in a subpackage without changing that loader.

Reading is inline below rather than factored out, because it is now two `scan_dataset` calls
selected by `evidence_format` -- a settings key, so translating it is this module's job. The recipe
takes a frame and knows nothing about storage, so a future per-datasource module inherits no
reading from it: one generating its evidence in polars reads nothing, and one generating in spark
reads its own parquet back through `scan_dataset` directly, with no format dispatch to reuse.

The registry lookup happens HERE, not in the recipe. `EXPRESSIONS` is keyed by `datasource_id` --
NOT taken from `settings['score_expression']` / `settings['direction_on_*_expression']`, even though
config.yaml still carries those spark-SQL strings for every `evidence_postprocess_*` step. The
compiler that used to translate them (`pts.transformers.utils.spark_sql`) has been deleted; the
strings are inert leftovers until a later change strips them from config.yaml for every datasource
at once. As per-datasource modules take over, each will supply its own expressions and this lookup
shrinks with the registry.
"""

from __future__ import annotations

from typing import Any

from loguru import logger
from otter.config.model import Config

from pts.transformers.evidence.expressions import EXPRESSIONS
from pts.transformers.evidence.postprocess import EvidencePostprocessor, ValidationLuts
from pts.transformers.utils.dataset import scan_dataset, write_dataset
from pts.transformers.utils.validation_lut import build_disease_lut, build_publication_lut, build_target_lut


def evidence_postprocess(
    source: dict[str, str],
    destination: dict[str, str],
    settings: dict[str, Any],
    config: Config,
) -> None:
    """Harmonise, validate, date and score one datasource's evidence, splitting valid from failed.

    Args:
        source: `evidence_path`, `target_path`, `disease_path`, `publication_date_lut`.
        destination: `evidence` and `failed_evidence` output directories.
        settings: `datasource_id`, `evidence_format`, `unique_fields`, and optionally
            `excluded_biotypes` -- see the module docstring for why the scoring/direction
            expressions come from `EXPRESSIONS`, not `settings`.
        config: otter config; unused, required by the `Transform` task's transformer signature.

    Raises:
        KeyError: if `datasource_id` has no entry in `EXPRESSIONS`.
    """
    datasource_id = settings['datasource_id']
    logger.info(f'processing "{datasource_id}" evidence')
    try:
        expressions = EXPRESSIONS[datasource_id]
    except KeyError:
        msg = f'no score/direction expressions registered for datasource {datasource_id!r} in EXPRESSIONS'
        raise KeyError(msg) from None

    luts = ValidationLuts(
        disease=build_disease_lut(source['disease_path']),
        target=build_target_lut(source['target_path']),
        publication=build_publication_lut(source['publication_date_lut']),
    )

    postprocessor = EvidencePostprocessor(
        datasource_id=datasource_id,
        unique_fields=settings['unique_fields'],
        expressions=expressions,
        excluded_biotypes=settings.get('excluded_biotypes'),
    )

    evidence_path, evidence_format = source['evidence_path'], settings['evidence_format']
    if evidence_format == 'parquet':
        lf = scan_dataset(evidence_path)
    elif evidence_format == 'json':
        # `infer_schema_length=None` scans the WHOLE file rather than polars' default 100-row
        # sample, which silently drops a column that first appears later -- measured, a bounded
        # sample missed a real column of `cosmic.json.gz`. Nothing else is pinned: dtypes are
        # inferred and then cast by harmonisation, which already covers every column
        # `evidence_schema` knows about, so pinning a schema here would only duplicate it.
        lf = scan_dataset(evidence_path, format='ndjson', infer_schema_length=None)
    else:
        # Only these two occur in config.yaml, so anything else is a config error rather than a
        # case to fall through to one reader or the other silently.
        msg = f'unrecognised evidence_format {evidence_format!r} for {evidence_path!r}, expected "parquet" or "json"'
        raise ValueError(msg)

    processed = postprocessor.run(lf, luts)

    # Two independent writes, not a persist-then-split like the pyspark implementation:
    # `.collect()`-ing once and splitting in memory doesn't bound memory, and the largest published
    # output (europepmc) is 9.03 GiB. This recomputes the upstream chain (LUT joins, hashing,
    # scoring, direction-of-effect) twice; an accepted trade for bounded memory, to revisit if
    # europepmc proves slow in practice.
    logger.info(f'writing valid evidence to {destination["evidence"]}')
    write_dataset(processed.valid, destination['evidence'])
    logger.info(f'writing failed evidence to {destination["failed_evidence"]}')
    write_dataset(processed.invalid, destination['failed_evidence'])
