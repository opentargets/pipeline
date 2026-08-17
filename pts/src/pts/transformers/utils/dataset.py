"""Shared dataset reading and writing for transformers.

One place that knows how a dataset is laid out on storage. Before this, `pts` read parquet five
different ways and wrote it nineteen times in three different styles, which is how the release
came to ship three compression codecs and how the `_`-prefix guard came to exist in exactly one
module.

`StorageHandle` resolves and lists; polars always receives URI STRINGS, never opened file
objects. That distinction is not stylistic: polars reads GCS natively with projection pushdown
(verified against a real bucket), but handed a file object it must materialise the whole thing --
measured 3.49 GiB peak RSS through a python file handle against 1.37 GiB through a path on
`eva.json.gz` (4,126,114 rows). On GCS a file object downloads the entire object before a single
column is read.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import polars as pl
from loguru import logger
from otter.storage.synchronous.handle import StorageHandle

#: Target size per written part, measured against the IN-MEMORY frame rather than the compressed
#: output -- so the on-disk size of a part depends on how well that dataset compresses. Calibrated
#: on the WORST measured on-disk/in-memory ratio (0.1756, evidence_clinical_precedence) rather
#: than the best (0.1220, evidence_eva), because calibrating on the best lands worse-compressing
#: data at ~360 MB. Measured range across six datasets: 171 MB to 246 MB per part.
#: Six datasets, all from partial local staging: a dataset compressing worse than 0.1756 will
#: still overshoot. This is a cap on typical data, not a guarantee.
_DEFAULT_TARGET_BYTES = 1_400_000_000

_GLOB_FOR_FORMAT: dict[str, str] = {
    'parquet': '*.parquet',
    # covers .json, .json.gz and .jsonl
    'ndjson': '*.json*',
}


def _parts(path: str, format: Literal['parquet', 'ndjson']) -> str | list[str]:
    """The file(s) making up one dataset, spark's `_`-prefix skip applied.

    Spark's readers silently skip files whose name starts with `_`, treating them as metadata
    (`_SUCCESS`, `_common_metadata`). A bare glob does not: measured, planting a
    `_hidden.parquet` in a real directory made it contribute rows. This lists and filters
    explicitly rather than relying on a glob to express the exclusion -- polars accepts
    `[^_]*.parquet` but NOT the shell-conventional `[!_]*.parquet`, and resting a correctness
    guard on undocumented glob-dialect trivia is how silent breakage happens later.

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

    Always a directory, never a single named file, so no dataset can outgrow its layout -- a
    single 9.03 GiB output is not an acceptable release artifact, and before this only the
    evidence step was protected from producing one.

    `pl.PartitionBy` never clears `path`; it only ever ADDS numbered parts to whatever is already
    there. So the destination is cleared first. Otter offers no delete for remote storage (there
    is no `delete`/`remove`/`unlink` anywhere in its storage layer), so this clearing is
    LOCAL-ONLY, which is sound because `release_uri` is per-release
    (`gs://open-targets-pre-data-releases/<release_name>`) and a production destination is
    therefore empty by construction. The case that actually recurs is a local re-run, which this
    covers. Re-running a release into an ALREADY-POPULATED remote URI would leave stale parts;
    that is a known limitation, stated here rather than silently relied upon.

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
