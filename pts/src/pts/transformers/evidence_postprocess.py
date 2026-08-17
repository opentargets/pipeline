"""Validate and post-process Open Targets Platform evidence.

Entry point for the `evidence_postprocess_*` config.yaml steps. Turns otter's
`source`/`destination`/`settings` into the parameters the recipe needs, reads the evidence, runs
`pts.transformers.evidence.postprocess`, and writes the two outputs. The processing itself lives in
the `pts.transformers.evidence` package.

It stays here rather than inside that package because otter resolves a transformer by importing
`pts.transformers.<name>` and taking the attribute of the same name, so a step module cannot live
in a subpackage without changing that loader.

Reading is inline: it is two `scan_dataset` calls selected by `evidence_format`, a settings key, so
translating it belongs to this module. The recipe itself performs no I/O.

The registry lookup also happens here, not in the recipe. `EXPRESSIONS` is keyed by
`datasource_id`; the `score_expression` and `direction_on_*_expression` strings config.yaml carries
for these steps are not read, and nothing translates them.
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
        # `infer_schema_length=None` scans the whole file. Polars' default samples the leading
        # rows and silently drops any column that first appears after it. No schema is pinned:
        # dtypes are inferred here and cast by harmonisation, which covers every column
        # `evidence_schema` knows about.
        lf = scan_dataset(evidence_path, format='ndjson', infer_schema_length=None)
    else:
        # Anything other than these two is a config error, not a case to fall through to one
        # reader or the other silently.
        msg = f'unrecognised evidence_format {evidence_format!r} for {evidence_path!r}, expected "parquet" or "json"'
        raise ValueError(msg)

    processed = postprocessor.run(lf, luts)

    # Two independent writes rather than collecting once and splitting in memory, which would put
    # a whole datasource in memory at once. Each write recomputes the chain; bounded memory is
    # worth more here than the duplicated work.
    logger.info(f'writing valid evidence to {destination["evidence"]}')
    write_dataset(processed.valid, destination['evidence'])
    logger.info(f'writing failed evidence to {destination["failed_evidence"]}')
    write_dataset(processed.invalid, destination['failed_evidence'])
