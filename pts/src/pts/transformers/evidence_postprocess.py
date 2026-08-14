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

import bz2
from typing import IO, Any

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


def _open_evidence_bytes(path: str) -> IO[bytes]:
    """Binary content of one json evidence part, decompressed if it is bzip2.

    polars' ndjson reader decompresses gzip transparently -- relied on everywhere else in this
    module via a plain `StorageHandle(path).open()` -- but not bzip2: fed a `.bz2` file, it does
    not refuse, it silently reads the still-compressed bytes as if they were plain content and
    raises `ComputeError: stream did not contain valid UTF-8`. `expression_atlas`'s evidence
    source (`input/evidence/atlas.json.bz2`) is bzip2, so this decompresses explicitly, keyed off
    the actual suffix rather than assuming atlas is the only `.bz2` source there will ever be.
    """
    raw = StorageHandle(path).open('rb')
    return bz2.open(raw, 'rb') if path.endswith('.bz2') else raw


def _json_schema(path: str) -> dict[str, Any]:
    """The schema to pin for one json evidence part: every column it actually carries, typed.

    A full `schema=load_spark_schema_as_polars('evidence.json')` pin does not just constrain
    dtypes, it MATERIALISES every one of the schema's 109 fields whether or not the source
    carries it -- measured: `reactome.json.gz`'s 12 real columns became 109 that way, filled with
    ~97 spurious all-null columns spark never produced (the published `evidence_reactome` output
    has 19 -- reactome's 12 plus `id`/`diseaseId`/etc. the `Evidence` chain adds on top). Spark's
    own json reader infers only the columns actually present, and `_harmonise_to_schema` only
    casts columns that are in both the frame and the target schema -- it never adds a schema
    field the frame lacks at the top level (that only happens one level down, inside a struct;
    see `_harmonise_expr`) -- so a wholesale materialise-every-field pin has no spark equivalent
    to justify it.

    The fix discovers the columns actually present with a FULL-file scan
    (`infer_schema_length=None`), not a bounded sample: a bounded 5,000-row sample measurably
    missed one of `cosmic.json.gz`'s real 9 columns in a real run here, the same class of defect
    `validation_lut.LITERATURE_SCHEMA` exists to avoid. Each discovered column is then typed with
    evidence.json's dtype where the column is one of its fields -- so casting downstream still
    lands on evidence.json's type, and the leading-rows dtype bug can't recur for any column this
    measurement covers -- or the inferred dtype otherwise, mapping an inferred Null (an all-null
    column with no evidence.json field to borrow a type from) to String.

    Args:
        path: location of one json evidence part.

    Returns:
        `{column: dtype}` for exactly the columns `path` carries.
    """
    inferred = pl.scan_ndjson(_open_evidence_bytes(path), infer_schema_length=None).collect_schema()
    target_schema = load_spark_schema_as_polars('evidence.json')
    return {
        name: target_schema.get(name, dtype if dtype != pl.Null else pl.String) for name, dtype in inferred.items()
    }


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

    return pl.concat([
        pl.read_ndjson(_open_evidence_bytes(part), schema=_json_schema(part)).lazy() for part in _json_parts(path)
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
