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
from pathlib import Path
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


def _decompress_bz2(path: str) -> bytes:
    """The fully decompressed bytes of one bzip2 json evidence part.

    polars' ndjson reader decompresses gzip transparently -- relied on for every other json
    source via a plain path string, see `_scan_json_part` -- but not bzip2: fed a `.bz2` source,
    it does not refuse, it silently reads the still-compressed bytes as if they were plain content
    and raises `ComputeError: stream did not contain valid UTF-8`. `expression_atlas`'s evidence
    source (`input/evidence/atlas.json.bz2`) is bzip2, so this decompresses explicitly, keyed off
    the actual suffix rather than assuming atlas is the only `.bz2` source there will ever be.

    Returns plain `bytes`, not a stream: `pl.scan_ndjson` accepts a `bytes` source directly and
    reuses it across two separate calls with no reset/seek bookkeeping (unlike a file-like
    object, which a first read would leave exhausted), so `_scan_json_part` can pin a schema and
    then scan the same bytes without decompressing bzip2's CPU-heavy stream twice.
    """
    handle = StorageHandle(path).open('rb')
    try:
        return bz2.decompress(handle.read())
    finally:
        handle.close()


def _json_schema(source: str | bytes) -> dict[str, Any]:
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
        source: one json evidence part's path (plain or gzip -- polars decompresses gzip
            natively from a path, which is also what keeps the read lazy, see `_scan_json_part`),
            or the already-decompressed bytes of a bzip2 part.

    Returns:
        `{column: dtype}` for exactly the columns `source` carries, in ALPHABETICAL order (see the
        sort below for why).
    """
    inferred = pl.scan_ndjson(source, infer_schema_length=None).collect_schema()
    target_schema = load_spark_schema_as_polars('evidence.json')
    # Sorted by name, not `inferred`'s own order: passing this dict as `pl.scan_ndjson(schema=...)`
    # (`_scan_json_part`) also fixes the resulting frame's COLUMN ORDER, not just its dtypes -- and
    # that order must be alphabetical to match spark. Spark's json reader sorts inferred columns
    # alphabetically; polars' keeps the order columns first appear in the file. Measured on a
    # one-line `{"zebra":..,"alpha":..,"mango":..}` fixture: spark yields
    # `['alpha', 'mango', 'zebra']`, polars (pre-sort) yields `['zebra', 'alpha', 'mango']`. Left
    # unsorted, that source order propagates through `_harmonise_to_schema` (order-preserving
    # `with_columns`) all the way to `validate_uniqueness`'s content hash, which iterates the
    # frame's columns in whatever order they are in -- an unsorted reader silently picks a
    # different surviving row among same-`id` duplicates than spark did.
    return {
        name: target_schema.get(name, inferred[name] if inferred[name] != pl.Null else pl.String)
        for name in sorted(inferred)
    }


def _scan_json_part(part: str) -> pl.LazyFrame:
    """Lazily scan one json evidence part, its own schema discovered and pinned.

    Passes `part`'s path straight to `pl.scan_ndjson` rather than an opened file object: doing so
    lets polars' Rust engine own the read (gzip decompressed natively, streamed rather than
    materialised) instead of going through `pl.read_ndjson` on a Python file handle, which is
    eager regardless of a trailing `.lazy()` -- measured on `eva.json.gz` (4,126,114 rows, the
    largest json evidence source), the eager path costs 3.49 GiB peak RSS against 1.37 GiB here,
    same row count. bzip2 has no such native support (`_decompress_bz2`), so a `.bz2` part is
    decompressed once and that same in-memory buffer is reused for both the schema pass and the
    actual scan -- not re-opened/re-decompressed per call, which would double bzip2's CPU cost for
    no benefit.

    Args:
        part: location of one json evidence part; see `_json_parts`.

    Returns:
        LazyFrame of `part`'s raw evidence, in its source columns/dtypes.
    """
    source = _decompress_bz2(part) if part.endswith('.bz2') else part
    return pl.scan_ndjson(source, schema=_json_schema(source))


def _parquet_parts(path: str) -> str | list[str]:
    """The parquet part(s) making up one evidence source, spark's `_`-prefix skip applied.

    Spark's parquet reader silently skips files whose name starts with `_` -- it treats them as
    metadata (`_SUCCESS`, `_common_metadata`), not data. A bare `f'{path}/*.parquet'` glob does
    not: measured, planting a `_hidden.parquet` in a real directory makes it contribute rows to a
    `pl.scan_parquet` read here, a real divergence from spark. This lists and filters explicitly
    so a stray metadata-shaped file can't silently join the evidence.

    Args:
        path: location of the evidence -- a single parquet file or a directory of parts.

    Returns:
        `path` unchanged when it is a single file, otherwise its sorted non-`_`-prefixed part
        locations.

    Raises:
        ValueError: if `path` is a directory with no non-`_`-prefixed `*.parquet` part. Left
            unguarded, `pl.scan_parquet([])` (`_read_evidence`) raises its own
            `ComputeError: empty input: paths: []` -- legible enough once you know to look here,
            but `_json_parts` already raises a clear `ValueError` for the equivalent json case, so
            this matches it rather than leaving the two readers inconsistent.
    """
    handle = StorageHandle(path)
    if not handle.stat().is_dir:
        return path
    parts = sorted(part for part in handle.glob('*.parquet') if not Path(part).name.startswith('_'))
    if not parts:
        raise ValueError(f'no parquet files found in {path}')
    return parts


def _read_evidence(path: str, evidence_format: str) -> pl.LazyFrame:
    """Read one datasource's raw evidence, before harmonisation to the evidence.json schema.

    Args:
        path: location of the evidence -- a directory of parquet parts or a single
            (possibly gzip/bz2-compressed) ndjson file; see `_json_parts`/`_parquet_parts`.
        evidence_format: `settings['evidence_format']`, `'parquet'` or `'json'`.

    Returns:
        LazyFrame of the raw evidence, in its source columns/dtypes.

    Raises:
        ValueError: if `evidence_format` is neither `'parquet'` nor `'json'` -- only those two
            occur in config.yaml today, so an unrecognised value is a config error, not a case
            to fall through to the json reader silently.
    """
    if evidence_format == 'parquet':
        return pl.scan_parquet(_parquet_parts(path))
    if evidence_format == 'json':
        # how='diagonal': spark's json reader unions differing per-part schemas rather than
        # demanding they match. Every evidence_postprocess step today points at a single json
        # file (`_json_parts` returns one part, so this never actually unions), but the default
        # how='vertical' raises the moment a step's source is a multi-part directory with any
        # schema drift between parts -- InvalidOperationError on a differing column set, ShapeError
        # on a differing column order -- so 'diagonal' is set now rather than left to be found the
        # hard way once a multi-part json step is wired.
        return pl.concat([_scan_json_part(part) for part in _json_parts(path)], how='diagonal')
    msg = f'unrecognised evidence_format {evidence_format!r} for {path!r}, expected "parquet" or "json"'
    raise ValueError(msg)


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
