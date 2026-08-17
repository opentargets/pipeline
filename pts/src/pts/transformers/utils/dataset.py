"""Shared dataset reading and writing for transformers.

One place that knows how a dataset is laid out on storage: which files make up a dataset, which
compression it is written with, and how large a part may get.

`StorageHandle` is used inconsistently here, on purpose, and the rule is worth stating because
the same module globs through it but unlinks around it:

* **Listing and stat go through it.** Deciding whether a location is one file or a directory of
  parts, and enumerating those parts, needs to work on every backend. That is what otter's
  abstraction is for.
* **Reads deliberately do not.** Polars receives URI STRINGS, never opened file objects. It reads
  cloud storage natively and pushes projections down to the file; handed a file object it can only
  materialise the whole thing, which on remote storage downloads every byte before a single column
  is read. Routing reads through the abstraction would cost more than it buys.
* **Deleting cannot.** otter exposes no remote delete, so clearing a destination falls back to
  `pathlib`, which silently does nothing on a remote path. That is a gap in the abstraction rather
  than a choice, and it is why `write_dataset` warns instead of pretending to have cleared.

So: use it where it adds reach, avoid it where it removes capability, and note where it simply
cannot help.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import polars as pl
from loguru import logger
from otter.storage.synchronous.handle import StorageHandle

#: Target size per written part. `pl.PartitionBy` sizes against the IN-MEMORY frame, not the
#: compressed output, so the resulting file size depends on how well a dataset compresses.
#: Calibrated against the worst compression ratio observed across the release rather than the
#: best, so that poorly-compressing datasets stay near the target instead of overshooting it --
#: which makes this a cap on typical data, not a guarantee. Aim is roughly 250 MB per file.
_DEFAULT_TARGET_BYTES = 1_400_000_000

_GLOB_FOR_FORMAT: dict[str, str] = {
    'parquet': '*.parquet',
    # covers .json, .json.gz and .jsonl
    'ndjson': '*.json*',
}


def _parts(path: str, format: Literal['parquet', 'ndjson']) -> str | list[str]:
    """The file(s) making up one dataset, spark's `_`-prefix skip applied.

    Spark's readers silently skip files whose name starts with `_`, treating them as metadata
    (`_SUCCESS`, `_common_metadata`); a bare glob does not, so a metadata-shaped file would
    contribute rows. This lists and filters explicitly rather than expressing the exclusion as a
    glob: polars accepts `[^_]*.parquet` but not the shell-conventional `[!_]*.parquet`, and a
    correctness guard resting on glob-dialect trivia breaks quietly.

    Args:
        path: the dataset location -- a single file or a directory of parts. Already absolute;
            `Transform.run` calls `make_absolute` before invoking a transformer.
        format: which glob to list a directory with.

    Returns:
        `path` unchanged when it is a single file, otherwise its sorted, non-`_`-prefixed parts.

    Raises:
        ValueError: if `path` is a directory containing no matching, non-`_`-prefixed file.
    """
    handle = StorageHandle(path)
    if not handle.stat().is_dir:
        return path
    pattern = _GLOB_FOR_FORMAT[format]
    parts = sorted(part for part in handle.glob(pattern) if not Path(part).name.startswith('_'))
    if not parts:
        raise ValueError(f'no {pattern} files found in {path}')
    return parts


def scan_dataset(
    path: str,
    *,
    format: Literal['parquet', 'ndjson'] = 'parquet',
    schema: Mapping[str, Any] | None = None,
) -> pl.LazyFrame:
    """Lazily read one dataset, whether it is a single file or a directory of parts.

    Always lazy: callers add `.collect()` where they need a `DataFrame`, so there is no second
    eager function to keep in sync.

    Args:
        path: the dataset location, already absolute.
        format: `'parquet'` or `'ndjson'`.
        schema: when given, pins dtypes AND column order. This function never invents a schema --
            a caller that needs one discovered from the data computes it and passes it in.

    Returns:
        LazyFrame of the dataset in its source columns and dtypes.

    Raises:
        ValueError: for an unrecognised `format`, a glob passed as `path`, a directory with no
            readable parts, or a schema passed with parquet format.
    """
    if format not in _GLOB_FOR_FORMAT:
        msg = f'unrecognised format {format!r} for {path!r}, expected "parquet" or "ndjson"'
        raise ValueError(msg)

    if any(char in path for char in '*?['):
        msg = (
            f'{path!r} looks like a glob; pass the containing directory instead. '
            'scan_dataset lists a directory itself and applies the _-prefix skip.'
        )
        raise ValueError(msg)

    if schema is not None and format == 'parquet':
        msg = f'schema is only applied to ndjson, but format is parquet for {path!r}'
        raise ValueError(msg)

    parts = _parts(path, format)
    if format == 'parquet':
        return pl.scan_parquet(parts)
    return pl.scan_ndjson(parts, schema=schema)


def write_dataset(
    frame: pl.LazyFrame | pl.DataFrame,
    path: str,
    *,
    approximate_bytes_per_file: int = _DEFAULT_TARGET_BYTES,
) -> None:
    """Write one dataset as a directory of size-capped zstd parquet parts.

    Always a directory, never a single named file, so no dataset can outgrow its layout as it
    grows.

    `pl.PartitionBy` never clears `path`; it only ever ADDS numbered parts to whatever is already
    there, so the destination is cleared first. Without that, a re-run producing fewer or
    differently-sized parts leaves the previous run's files behind, and every consumer globbing
    `*.parquet` reads them as current data.

    That clearing is LOCAL-ONLY: otter exposes no delete for remote storage. It is sound in
    production because each release writes to its own URI, so the destination starts empty -- but
    a retry into an already-populated remote destination can leave parts from the previous
    attempt. Hence the warning below, rather than a silent no-op.

    Args:
        frame: the data to write; a `DataFrame` is made lazy so there is one code path.
        path: destination directory, used as given -- never a parent or a derived path.
        approximate_bytes_per_file: target part size, measured against the IN-MEMORY frame. See
            `_DEFAULT_TARGET_BYTES` for the calibration and its limits.

    Raises:
        ValueError: if `path` exists and is not a directory. Every configured destination is a
            directory; a file there means the layout is not what this expects, so it refuses
            rather than deleting something it does not understand.
    """
    if '://' in path:
        logger.warning(
            f'{path} is remote; stale parts cannot be cleared (otter has no remote delete). '
            'A retry into a populated destination can leave parts from the previous attempt.'
        )

    directory = Path(path)
    if directory.exists():
        if not directory.is_dir():
            msg = f'expected destination {path!r} to be a directory (or not exist yet), found a file'
            raise ValueError(msg)
        for part in directory.glob('*.parquet'):
            part.unlink()

    lf = frame.lazy() if isinstance(frame, pl.DataFrame) else frame
    lf.sink_parquet(
        pl.PartitionBy(path, approximate_bytes_per_file=approximate_bytes_per_file),
        compression='zstd',
    )
