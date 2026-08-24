"""Reading dataset statistics out of a release bucket.

Everything here is metadata: object listings and parquet footers. No dataset is ever
scanned, so comparing a whole release costs `O(files)` requests rather than the cost of
reading it.

The bucket is injected rather than constructed, matching `journal.py`, so the statistics
logic is testable without credentials. The footer reader is injected for the same reason
and at the same seam: it is the only part that must talk to `pyarrow`.

Spark leaves `_SUCCESS` markers and `.crc` sidecars beside the data it writes, and both
appear in a GCS listing even where a local `Path.glob` would skip them. Counting them as
data files would inflate every partition count and, worse, make an otherwise clean parquet
dataset look like a mixed-format one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel, Field

from orchestration.supervisor.diff import DatasetDiff, Side, compare_schemas

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

PARQUET_SUFFIX = '.parquet'


class Footer(BaseModel):
    """What one parquet file's footer tells us.

    Args:
        rows: The file's row count, read from its metadata.
        column_types: Column name to stringified type.
    """

    rows: int
    column_types: dict[str, str] = Field(default_factory=dict)


class DatasetStats(BaseModel):
    """One dataset's statistics, on one side of a comparison.

    Args:
        rows: Total rows across every file, or None when the format has no footer.
        total_bytes: Total size of the data files.
        files: How many data files, which catches a repartition that leaves the
            total size unchanged.
        column_types: The schema, taken from the first data file. Empty when
            unavailable.
        countable: False when the dataset is not parquet, so `rows` is unavailable
            rather than zero.
    """

    rows: int | None = None
    total_bytes: int = 0
    files: int = 0
    column_types: dict[str, str] = Field(default_factory=dict)
    countable: bool = True


class Blob(Protocol):
    """The part of a GCS blob this module uses."""

    name: str
    size: int | None


class Bucket(Protocol):
    """The part of a GCS bucket this module uses."""

    def list_blobs(self, prefix: str) -> Iterable[Blob]:
        """List the objects under a prefix."""
        ...


def is_data_file(name: str) -> bool:
    """Whether an object is dataset content rather than a sidecar.

    Args:
        name: The full object name.

    Returns:
        False for spark's `_SUCCESS` markers and `.crc` sidecars, and for the
        directory placeholder objects a prefix listing also returns.
    """
    basename = name.rsplit('/', 1)[-1]
    return bool(basename) and not basename.startswith(('_', '.'))


def read_stats(
    bucket: Bucket,
    prefix: str,
    read_footer: Callable[[str], Footer] | None = None,
) -> DatasetStats | None:
    """Read one dataset's statistics from a bucket.

    The prefix is anchored with a trailing `/` before listing. Without it,
    `output/disease` would also match `output/disease_hpo` — both real datasets — and
    silently fold one into the other.

    Args:
        bucket: The bucket to read.
        prefix: The dataset's full object prefix, e.g. `my-run/output/disease`.
        read_footer: Reads one parquet file's footer. Injected so the statistics
            logic is testable without credentials; None disables row counting,
            which is what a caller wanting sizes only should pass.

    Returns:
        The dataset's statistics, or None when no data file exists under the prefix,
        which is how an absent dataset is distinguished from an empty one.
    """
    blobs = [b for b in bucket.list_blobs(prefix=prefix.rstrip('/') + '/') if is_data_file(b.name)]
    if not blobs:
        return None

    blobs.sort(key=lambda b: b.name)
    total_bytes = sum(b.size or 0 for b in blobs)
    countable = all(b.name.endswith(PARQUET_SUFFIX) for b in blobs)

    if not countable or read_footer is None:
        return DatasetStats(total_bytes=total_bytes, files=len(blobs), countable=countable)

    rows = 0
    column_types: dict[str, str] = {}
    for blob in blobs:
        footer = read_footer(blob.name)
        rows += footer.rows
        if not column_types:
            column_types = footer.column_types
    return DatasetStats(
        rows=rows,
        total_bytes=total_bytes,
        files=len(blobs),
        column_types=column_types,
        countable=True,
    )


def compare(dataset: str, run: DatasetStats | None, reference: DatasetStats | None) -> DatasetDiff:
    """Assemble one dataset's diff from both sides' statistics.

    Args:
        dataset: The dataset's path relative to the release root, carrying its
            namespace, e.g. `output/disease`.
        run: Statistics from the run, or None when the dataset is absent there.
        reference: Statistics from the reference release, or None when absent.

    Returns:
        The comparison. Schemas are compared only when both sides exist, since a
        one-sided dataset has no columns to have changed.

    Raises:
        ValueError: If both sides are None, which means the caller is comparing a
            dataset that exists nowhere and would produce a diff describing nothing.
    """
    if run is None and reference is None:
        raise ValueError(f'{dataset!r} is absent from both sides, so there is nothing to compare')

    side: Side = 'both' if run and reference else 'run_only' if run else 'reference_only'
    columns = compare_schemas(run.column_types, reference.column_types) if run and reference else []
    return DatasetDiff(
        dataset=dataset,
        side=side,
        run_rows=run.rows if run else None,
        reference_rows=reference.rows if reference else None,
        run_bytes=run.total_bytes if run else 0,
        reference_bytes=reference.total_bytes if reference else 0,
        run_files=run.files if run else 0,
        reference_files=reference.files if reference else 0,
        columns=columns,
        countable=(run is None or run.countable) and (reference is None or reference.countable),
    )


def footer_reader(bucket_name: str) -> Callable[[str], Footer]:
    """Build a footer reader backed by pyarrow's GCS filesystem.

    Args:
        bucket_name: The bucket the object names are relative to.

    Returns:
        A callable reading one object's parquet footer. Constructed lazily so that
        importing this module needs neither pyarrow nor credentials.
    """
    import pyarrow.fs as pa_fs
    import pyarrow.parquet as pq

    from orchestration.supervisor.diff import schema_of

    filesystem = pa_fs.GcsFileSystem()

    def read(name: str) -> Footer:
        metadata = pq.read_metadata(f'{bucket_name}/{name}', filesystem=filesystem)
        return Footer(rows=metadata.num_rows, column_types=schema_of(metadata.schema.to_arrow_schema()))

    return read
