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

from pathlib import Path
from typing import Any

import polars as pl
from loguru import logger
from otter.config.model import Config
from otter.storage.synchronous.handle import StorageHandle

from pts.schemas.evidence import evidence_schema
from pts.transformers.utils.evidence import Evidence
from pts.transformers.utils.evidence_expressions import EXPRESSIONS
from pts.transformers.utils.validation_lut import build_disease_lut, build_publication_lut, build_target_lut

# `pl.PartitionBy.approximate_bytes_per_file` sizes to the estimated IN-MEMORY frame, not the
# compressed on-disk parquet -- the two differ by roughly 8x for evidence. Calibrated on
# `evidence_eva` (3,998,459 rows, the largest staged local output): its single-file parquet is
# 254.8 MB on disk against a 2,085.8 MB `DataFrame.estimated_size()`, a 0.122 on-disk/in-memory
# ratio. Confirmed by replicating that frame 5x and sinking through `pl.PartitionBy` at this
# setting: the resulting parts land at 294-301 MB on disk (bar the final, smaller remainder part),
# matching the 300 MB target. Revisit this constant if evidence's column mix (and so its
# compression ratio) changes materially.
_TARGET_BYTES_PER_FILE = 2_450_000_000


def _json_schema(source: str) -> dict[str, Any]:
    """The schema to pin for one json evidence part: every column it actually carries, typed.

    A full `schema=evidence_schema` pin does not just constrain
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
    `evidence_schema`'s dtype where the column is one of its fields -- so casting downstream still
    lands on that type, and the leading-rows dtype bug can't recur for any column this
    measurement covers -- or the inferred dtype otherwise, mapping an inferred Null (an all-null
    column with no `evidence_schema` field to borrow a type from) to String.

    Args:
        source: one json evidence part's path (plain or gzip -- polars decompresses gzip
            natively from a path, which is also what keeps the read lazy, see `_scan_json_part`).

    Returns:
        `{column: dtype}` for exactly the columns `source` carries, in ALPHABETICAL order (see the
        sort below for why).
    """
    inferred = pl.scan_ndjson(source, infer_schema_length=None).collect_schema()
    target_schema = evidence_schema
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


def _scan_json_part(path: str) -> pl.LazyFrame:
    """Lazily scan the json evidence source, its own schema discovered and pinned.

    Passes `path` straight to `pl.scan_ndjson` rather than an opened file object: doing so
    lets polars' Rust engine own the read (gzip decompressed natively, streamed rather than
    materialised) instead of going through `pl.read_ndjson` on a Python file handle, which is
    eager regardless of a trailing `.lazy()` -- measured on `eva.json.gz` (4,126,114 rows, the
    largest json evidence source), the eager path costs 3.49 GiB peak RSS against 1.37 GiB here,
    same row count.

    Args:
        path: location of the json evidence source; a single file, see `_read_evidence`.

    Returns:
        LazyFrame of the raw evidence, in its source columns/dtypes.
    """
    return pl.scan_ndjson(path, schema=_json_schema(path))


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
            but `_read_evidence` already raises a clear `ValueError` when a json source is a
            directory at all, so this matches it rather than leaving the two readers inconsistent.
    """
    handle = StorageHandle(path)
    if not handle.stat().is_dir:
        return path
    parts = sorted(part for part in handle.glob('*.parquet') if not Path(part).name.startswith('_'))
    if not parts:
        raise ValueError(f'no parquet files found in {path}')
    return parts


def _read_evidence(path: str, evidence_format: str) -> pl.LazyFrame:
    """Read one datasource's raw evidence, before harmonisation to `evidence_schema`.

    Args:
        path: location of the evidence -- a directory of parquet parts (see `_parquet_parts`) or
            a single (possibly gzip-compressed) ndjson file.
        evidence_format: `settings['evidence_format']`, `'parquet'` or `'json'`.

    Returns:
        LazyFrame of the raw evidence, in its source columns/dtypes.

    Raises:
        ValueError: if `evidence_format` is neither `'parquet'` nor `'json'` -- only those two
            occur in config.yaml today, so an unrecognised value is a config error, not a case
            to fall through to the json reader silently. Also raised if `evidence_format` is
            `'json'` and `path` is a directory: every json evidence source in config.yaml
            (`input/evidence/*.json.gz`) is a single file, unlike the parquet sources'
            directory-of-parts shape, so a directory here is a config error too, refused rather
            than silently globbed or left to fail inside polars with a confusing error.
    """
    if evidence_format == 'parquet':
        return pl.scan_parquet(_parquet_parts(path))
    if evidence_format == 'json':
        if StorageHandle(path).stat().is_dir:
            msg = f'json evidence must be a single file, got a directory: {path!r}'
            raise ValueError(msg)
        return _scan_json_part(path)
    msg = f'unrecognised evidence_format {evidence_format!r} for {path!r}, expected "parquet" or "json"'
    raise ValueError(msg)


def _write_partitioned(lf: pl.LazyFrame, path: str) -> None:
    """Sink `lf` into `path` as a directory of size-capped parquet parts, first clearing it.

    `pl.PartitionBy` writes a directory of size-capped parts rather than one file -- a single
    9.03 GiB europepmc file is not an acceptable release artifact. It is marked UNSTABLE in
    polars 1.41.2; checked empirically that it raises no runtime warning at either construction
    or sink time on this version, so nothing is suppressed here -- if a future polars upgrade
    starts emitting one, handle it deliberately rather than silencing it. A datasource under
    `_TARGET_BYTES_PER_FILE` in memory still gets exactly one part; nothing extra is needed for
    that, it falls out of `approximate_bytes_per_file` sizing (see its calibration comment).

    `pl.PartitionBy` itself never clears `path` -- it only ever adds numbered parts to whatever
    is already there. The pyspark implementation this replaced used `.write.mode('overwrite')`,
    which does clear the destination; that overwrite semantics is reproduced by hand here because
    nothing else in the stack provides it. `otter`'s `check_destination(path, delete=True)`
    (`otter/util/fs.py`) does NOT cover this case: it only unlinks `path` when `path.is_file()`,
    and silently does nothing for a directory -- which `path` always is here. Left unhandled, a
    stale part survives untouched alongside the new ones (a leftover from an earlier, differently
    sized run, or -- before the switch to `pl.PartitionBy` -- the old single-file layout), and
    every consumer globbing `*.parquet` under `path` (release metrics, croissant, spark) silently
    reads and counts it as current data.

    Args:
        lf: the frame to write.
        path: destination directory -- `destination['evidence']` or `destination['failed_evidence']`,
            used as given, never a parent or a derived path.

    Raises:
        ValueError: if `path` already exists and is not a directory. Every configured destination
            is a directory; a file there means the layout is not what this expects, so it refuses
            rather than deleting something it does not understand.
    """
    directory = Path(path)
    if directory.exists():
        if not directory.is_dir():
            msg = f'expected evidence destination {path!r} to be a directory (or not exist yet), found a file'
            raise ValueError(msg)
        for part in directory.glob('*.parquet'):
            part.unlink()
    lf.sink_parquet(pl.PartitionBy(path, approximate_bytes_per_file=_TARGET_BYTES_PER_FILE))


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
    # implementation: `.collect()`-ing once and splitting in memory doesn't bound memory, and the
    # largest published output (europepmc) is 9.03 GiB. This recomputes the upstream chain (LUT
    # joins, hashing, scoring, direction-of-effect) twice; an accepted trade for bounded memory,
    # to revisit if europepmc proves slow in practice.
    logger.info(f'writing valid evidence to {destination["evidence"]}')
    _write_partitioned(processed.valid(), destination['evidence'])
    logger.info(f'writing failed evidence to {destination["failed_evidence"]}')
    _write_partitioned(processed.invalid(), destination['failed_evidence'])
