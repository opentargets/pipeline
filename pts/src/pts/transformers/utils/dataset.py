"""Shared dataset reading and writing for transformers.

One place that knows how a dataset is laid out on storage: which files make up a dataset, which
compression it is written with, and how large a part may get.

`StorageHandle` is used for some things here and deliberately not others, which is worth stating
because the mix looks arbitrary otherwise:

* **Listing and existence go through it.** Deciding whether a location is one file or a directory
  of parts, enumerating those parts, and checking whether a destination is already occupied all
  have to work on every backend. That is what otter's abstraction is for.
* **Reads deliberately do not.** Polars receives URI STRINGS, never opened file objects. It reads
  cloud storage natively and pushes projections down to the file; handed a file object it can only
  materialise the whole thing, which on remote storage downloads every byte before a single column
  is read. Routing reads through the abstraction would cost more than it buys.
* **Deleting is not attempted at all**, because otter exposes no remote delete. Anything built on
  `pathlib` would work locally and be a silent no-op on the cloud destinations that matter, so
  `write_dataset` refuses to write into an occupied destination rather than clearing it. That
  behaves identically on every backend, which a clear could not.

So: use it where it adds reach, avoid it where it removes capability, and design around what it
cannot do rather than half-doing it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import polars as pl
from otter.storage.synchronous.handle import StorageHandle
from otter.util.errors import NotFoundError

#: Target size per written part. `pl.PartitionBy` sizes against the IN-MEMORY frame, not the
#: compressed output, so the resulting file size depends on how well a dataset compresses.
#: Calibrated against the worst compression ratio observed across the release rather than the
#: best, so that poorly-compressing datasets stay near the target instead of overshooting it --
#: which makes this a cap on typical data, not a guarantee. Aim is roughly 250 MB per file.
_DEFAULT_TARGET_BYTES = 1_400_000_000

Format = Literal['parquet', 'ndjson', 'csv', 'tsv']

#: Glob used to enumerate a directory for each format.
_GLOBS: dict[str, str] = {
    'parquet': '*.parquet',
    # covers .json, .json.gz and .jsonl
    'ndjson': '*.json*',
    'csv': '*.csv*',
    'tsv': '*.tsv*',
}

#: The delimited formats, which differ only in separator and accept caller parsing options.
_SEPARATORS: dict[str, str] = {'csv': ',', 'tsv': '\t'}


def scan_dataset(
    path: str,
    *,
    format: Format = 'parquet',
    schema: Mapping[str, Any] | None = None,
    **options: Any,
) -> pl.LazyFrame:
    """Lazily read one dataset, whether it is a single file or a directory of parts.

    Always lazy: callers add `.collect()` where they need a `DataFrame`, so there is no second
    eager function to keep in sync.

    A directory is enumerated through `StorageHandle` rather than handed to polars as a glob
    string, so the part list is explicit and sorted and does not depend on a glob dialect. Files
    whose name starts with `_` are NOT excluded: spark writes `_SUCCESS` and `_common_metadata`
    without a data extension, so the per-format globs already miss them, and we control how these
    datasets are produced.

    `csv` and `tsv` differ only in separator. Both forward `**options` to `pl.scan_csv`, because
    delimited sources vary in ways parquet does not -- headers, quoting, comment prefixes,
    explicit column names -- and those belong to the caller, not here. `ndjson` forwards them to
    `pl.scan_ndjson` for the same reason; `infer_schema_length` in particular matters, since its
    default of 100 rows silently drops a column that first appears later in the file.

    Parquet takes no options: it carries its own schema, so there is nothing for a caller to
    describe.

    Args:
        path: the dataset location, already absolute.
        format: `'parquet'`, `'ndjson'`, `'csv'` or `'tsv'`.
        schema: when given, pins dtypes AND column order. This function never invents a schema --
            a caller that needs one discovered from the data computes it and passes it in. Not
            supported for parquet, which carries its own.
        **options: forwarded to the underlying polars reader for `ndjson`, `csv` and `tsv`.
            Rejected for parquet, where they would silently do nothing.

    Returns:
        LazyFrame of the dataset in its source columns and dtypes.

    Raises:
        ValueError: for an unrecognised `format`, a glob passed as `path`, a directory with no
            readable parts, a schema passed with parquet, or options passed to a format that
            cannot use them.
    """
    if format not in _GLOBS:
        msg = f'unrecognised format {format!r} for {path!r}, expected one of {sorted(_GLOBS)}'
        raise ValueError(msg)

    if any(char in path for char in '*?['):
        msg = (
            f'{path!r} looks like a glob; pass the containing directory instead. '
            'scan_dataset enumerates a directory itself.'
        )
        raise ValueError(msg)

    if schema is not None and format == 'parquet':
        msg = f'schema is only applied to ndjson, csv and tsv, but format is parquet for {path!r}'
        raise ValueError(msg)

    if options and format == 'parquet':
        msg = f'options {sorted(options)} are not forwarded for parquet, which carries its own schema'
        raise ValueError(msg)

    pattern = _GLOBS[format]
    handle = StorageHandle(path)
    if handle.stat().is_dir:
        parts: str | list[str] = sorted(handle.glob(pattern))
        if not parts:
            raise ValueError(f'no {pattern} files found in {path}')
    else:
        parts = path

    if format == 'parquet':
        return pl.scan_parquet(parts)
    if format == 'ndjson':
        return pl.scan_ndjson(parts, schema=schema, **options)
    return pl.scan_csv(parts, separator=_SEPARATORS[format], schema=schema, **options)


def scan_datasets(
    pattern: str,
    *,
    format: Format = 'parquet',
    **options: Any,
) -> pl.LazyFrame:
    """Lazily read every dataset whose directory name matches a glob, as one frame.

    `scan_dataset` refuses a glob because it enumerates a single directory itself. This is the
    dataset-level counterpart: the glob selects DIRECTORIES (`output/evidence_*`), and every
    matched directory is read and concatenated.

    Schemas are unioned rather than required to match. Most evidence datasets carry no
    `drugId` column at all, and the pyspark job relied on `mergeSchema=True` to null-fill it;
    `missing_columns='insert'` is the polars equivalent.

    Raises rather than returning an empty frame when nothing matches: an empty evidence frame
    would silently strip every drug from the disease index instead of failing the step.

    Args:
        pattern: dataset location containing a glob in its LAST segment, already absolute.
        format: `'parquet'`, `'ndjson'`, `'csv'` or `'tsv'`.
        **options: forwarded to `scan_dataset` for each matched dataset.

    Returns:
        LazyFrame of every matched dataset, vertically concatenated with a unioned schema.

    Raises:
        ValueError: if `pattern` has no glob, if the glob is not in the last segment, or if
            it matches no datasets.
    """
    parent, _, name_glob = pattern.rpartition('/')
    if not any(char in name_glob for char in '*?['):
        msg = f'{pattern!r} is not a glob; use scan_dataset for a single dataset'
        raise ValueError(msg)
    if any(char in parent for char in '*?['):
        msg = f'{pattern!r} globs above the dataset name; only the last segment may be a glob'
        raise ValueError(msg)

    directories = sorted(StorageHandle(parent).glob(name_glob))
    if not directories:
        msg = f'{pattern!r} matched no datasets'
        raise ValueError(msg)

    frames = [scan_dataset(directory, format=format, **options) for directory in directories]
    return pl.concat(frames, how='diagonal_relaxed')


def write_dataset(
    frame: pl.LazyFrame | pl.DataFrame,
    path: str,
    *,
    approximate_bytes_per_file: int = _DEFAULT_TARGET_BYTES,
) -> None:
    """Write one dataset as a directory of size-capped zstd parquet parts.

    Always a directory, never a single named file, so no dataset can outgrow its layout as it
    grows.

    REFUSES to write into anything that already exists. `pl.PartitionBy` never clears a
    destination -- it only ever ADDS numbered parts -- so writing into a populated directory
    leaves the previous run's files in place, and every consumer globbing `*.parquet` reads them
    as current data. A re-run producing fewer or differently-sized parts silently duplicates rows
    that way, which is worse than not running at all.

    Clearing instead of refusing was considered and rejected: otter exposes no delete for remote
    storage, so a clear could only ever work locally and would be a silent no-op on the cloud
    destinations that matter. Refusing behaves identically everywhere. There is no overwrite mode
    yet; clear the destination deliberately, or write somewhere new.

    Note what this costs locally: re-running a step twice in a row now fails on the second run
    until its destination is removed. In production each release writes to its own URI, so this
    fires only on a genuine retry into a populated destination -- exactly the case that used to
    corrupt quietly.

    Args:
        frame: the data to write; a `DataFrame` is made lazy so there is one code path.
        path: destination directory, used as given -- never a parent or a derived path.
        approximate_bytes_per_file: target part size, measured against the IN-MEMORY frame. See
            `_DEFAULT_TARGET_BYTES` for the calibration and its limits.

    Raises:
        ValueError: if anything already exists at `path`, file or directory.
    """
    try:
        StorageHandle(path).stat()
    except NotFoundError:
        pass
    else:
        msg = (
            f'{path!r} already exists; write_dataset never overwrites. `pl.PartitionBy` only adds '
            'parts, so writing here would leave the previous run behind. Remove it first.'
        )
        raise ValueError(msg)

    lf = frame.lazy() if isinstance(frame, pl.DataFrame) else frame
    lf.sink_parquet(
        pl.PartitionBy(path, approximate_bytes_per_file=approximate_bytes_per_file),
        compression='zstd',
    )
