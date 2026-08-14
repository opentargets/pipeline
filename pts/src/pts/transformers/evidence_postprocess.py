"""Validate and post-process Open Targets Platform evidence.

Polars entry point for the `evidence_postprocess_*` config.yaml steps, replacing
`pts.pyspark.evidence_postprocess`. Wires the lookup-table builders
(`pts.transformers.utils.validation_lut`) and the `Evidence` chain
(`pts.transformers.utils.evidence`) into one `Transform` task, in the same call order
the pyspark implementation used.

Score and direction-of-effect expressions come from
`pts.transformers.utils.evidence_expressions.EXPRESSIONS`, keyed by `datasource_id` --
NOT from `settings['score_expression']` / `settings['direction_on_*_expression']`, even
though config.yaml still carries those spark-SQL strings for every `evidence_postprocess_*`
step. The compiler that used to translate them (`pts.transformers.utils.spark_sql`) has
been deleted; the strings are inert leftovers until a later change strips them from
config.yaml for every datasource at once.
"""

from __future__ import annotations

from typing import Any

import polars as pl
from loguru import logger
from otter.config.model import Config
from otter.storage.synchronous.handle import StorageHandle

from pts.transformers.utils.evidence import Evidence
from pts.transformers.utils.evidence_expressions import EXPRESSIONS
from pts.transformers.utils.schemas import load_spark_schema_as_polars
from pts.transformers.utils.validation_lut import build_disease_lut, build_publication_lut, build_target_lut


def _json_parts(path: str) -> list[str]:
    """The ndjson file(s) making up one evidence source, whatever shape `path` is.

    Extends `validation_lut._parts` to also accept a single file: config.yaml's json
    evidence sources (`input/evidence/*.json.gz`/`.bz2`) are always one file, but
    `StorageHandle(path).open()` raises `IsADirectoryError` on the parquet sources'
    shape (`intermediate/evidence/*.parquet`, a directory of parts), so this checks
    rather than assumes which one `path` is.

    Args:
        path: location of the json evidence source.

    Returns:
        `[path]` when it is a single file, otherwise its sorted part locations.

    Raises:
        ValueError: if `path` is a directory with no `*.json*` part.
    """
    handle = StorageHandle(path)
    if not handle.stat().is_dir:
        return [path]
    parts = sorted(handle.glob('*.json*'))
    if not parts:
        raise ValueError(f'no json files found in {path}')
    return parts


def _read_evidence(path: str, evidence_format: str) -> pl.LazyFrame:
    """Read one datasource's raw evidence, before harmonisation to the evidence.json schema.

    Args:
        path: location of the evidence -- a directory of parquet parts or a single
            (possibly gzip/bz2-compressed) ndjson file; see `_json_parts`.
        evidence_format: `settings['evidence_format']`, `'parquet'` or `'json'`.

    Returns:
        LazyFrame of the raw evidence, in its source columns/dtypes.
    """
    if evidence_format == 'parquet':
        is_dir = StorageHandle(path).stat().is_dir
        return pl.scan_parquet(f'{path}/*.parquet' if is_dir else path)

    # A bare `pl.read_ndjson(path)` infers each column's dtype from a sample of the leading
    # rows, which -- under `ignore_errors=True` -- is the exact construct that silently
    # discarded 386,627 pmids elsewhere in this port (see validation_lut.LITERATURE_SCHEMA).
    # Pinning the full evidence.json schema removes that risk, but trades in a different one:
    # `schema=` (not `schema_overrides=`) restricts the read to exactly the named columns, so a
    # raw column the source carries but evidence.json doesn't would be silently dropped rather
    # than passed through for `Evidence.__post_init__`/`_harmonise_to_schema` to leave alone.
    # Checked, not assumed: every json evidence source staged for this release --
    # atlas (242,545 rows), cosmic (102,872), eva (4,126,114, the largest), reactome (11,492),
    # uniprot_literature (7,697), uniprot_variants (47,867), validation_lab (1,125) -- was
    # scanned end to end, and every raw column each one carries is already a field of
    # evidence.json, so the full-schema pin drops nothing for any of them today.
    target_schema = load_spark_schema_as_polars('evidence.json')
    return pl.concat([
        pl.read_ndjson(StorageHandle(part).open(), schema=target_schema).lazy() for part in _json_parts(path)
    ])


def evidence_postprocess(
    source: dict[str, str],
    destination: dict[str, str],
    settings: dict[str, Any],
    config: Config,
) -> None:
    """Harmonise, validate, date and score one datasource's evidence, splitting valid from failed.

    Args:
        source: `evidence_path`, `target_path`, `disease_path`, `publication_date_lut`.
        destination: `evidence` and `failed_evidence` output parquet file paths.
        settings: `datasource_id`, `evidence_format`, `unique_fields`, and optionally
            `excluded_biotypes` -- see the module docstring for why the scoring/direction
            expressions here come from `EXPRESSIONS`, not `settings`.
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

    disease_lut = build_disease_lut(source['disease_path'])
    target_lut = build_target_lut(source['target_path'])
    publication_lut = build_publication_lut(source['publication_date_lut'])

    lf = _read_evidence(source['evidence_path'], settings['evidence_format'])

    processed = (
        Evidence(lf)
        .validate_diseases(disease_lut)
        .validate_target(target_lut, settings.get('excluded_biotypes'))
        .validate_datasource(datasource_id)
        .assign_evidence_identifier(settings['unique_fields'])
        .validate_uniqueness()
        .resolve_publication_date(publication_lut)
        .resolve_evidence_date()
        .calculate_evidence_score(expressions.score)
        .assign_direction_on_trait(expressions.direction_on_trait)
        .assign_direction_on_target(expressions.direction_on_target, target_lut)
        .hash_long_variant_identifiers()
    )

    # Two independent sink_parquet calls, not a persist-then-split like the pyspark
    # implementation: polars 1.41.2 has no partitioned sink (`sink_parquet` takes one path),
    # so every dataset is one file, and the largest published output (europepmc) is 9.03 GiB --
    # too big to `.collect()` before splitting valid from invalid. This recomputes the upstream
    # chain (LUT joins, hashing, scoring, direction-of-effect) twice; an accepted trade for
    # bounded memory, to revisit if europepmc proves slow in practice.
    logger.info(f'writing valid evidence to {destination["evidence"]}')
    processed.valid().sink_parquet(destination['evidence'])
    logger.info(f'writing failed evidence to {destination["failed_evidence"]}')
    processed.invalid().sink_parquet(destination['failed_evidence'])
