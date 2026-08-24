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

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel, Field

from orchestration.supervisor.datasets import destinations_for
from orchestration.supervisor.diff import DatasetDiff, Side, compare_schemas
from orchestration.supervisor.step_identity import identify

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from typing import Any

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
    max_workers: int = 16,
) -> DatasetStats | None:
    """Read one dataset's statistics from a bucket.

    The prefix is anchored with a trailing `/` before listing. Without it,
    `output/disease` would also match `output/disease_hpo` — both real datasets — and
    silently fold one into the other.

    The schema is the first blob's, by name, unconditionally — not the first *non-empty*
    schema encountered. A first file with no columns is therefore reported as an empty
    schema rather than skipped in favour of a later file's. This is deliberate: a
    footer read concurrently cannot fall back to "whichever file happens to have
    columns" without letting completion order decide the schema, which is exactly the
    ambiguity the thread pool below must not reintroduce. In practice the case does
    not arise — a parquet file carries its schema even at zero rows, so `column_types`
    is empty only for a file with genuinely no columns at all.

    Args:
        bucket: The bucket to read.
        prefix: The dataset's full object prefix, e.g. `my-run/output/disease`.
        read_footer: Reads one parquet file's footer. Injected so the statistics
            logic is testable without credentials; None disables row counting,
            which is what a caller wanting sizes only should pass.
        max_workers: How many footers to read concurrently. Measured against the real
            release buckets: a single footer read costs ~0.34s, and one dataset alone
            (`output/association_overall_direct`) has 43 files, so a serial read of a
            release's 2,602 files across 66 datasets is 25-30 minutes. A thread pool
            gives 3.3x at this width — measured 8 -> 2.9x, 16 -> 3.3x, 32 -> 3.0x, with
            identical row totals at every width — which is why 16 is the default
            rather than a wider or unbounded pool.

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

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        # `Executor.map` yields results in the order the calls were made, i.e. blob
        # (name) order, regardless of which thread finishes first. The schema below
        # must come from `blobs[0]`, not from whichever footer happens to read fastest.
        footers = list(pool.map(read_footer, (blob.name for blob in blobs)))

    return DatasetStats(
        rows=sum(footer.rows for footer in footers),
        total_bytes=total_bytes,
        files=len(blobs),
        column_types=footers[0].column_types,
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


class Skipped(BaseModel):
    """What a comparison did not cover, so the report can say so rather than imply completeness.

    Args:
        steps_without_datasets: Steps declaring no release destination. This is the
            majority of steps — 68 of 125 — and is not an anomaly.
        stages_without_config: Steps whose stage has no local config, which is how
            gentropy's twelve steps are excluded; their destinations live in
            `dags/config/gentropy.yaml` and are not read here.
        datasets_absent_from_both: Destinations found in neither bucket, which usually
            means the step has not run yet.
        undeclared_in_buckets: Datasets present in one of the buckets that no step
            declares. The walk is driven by the configs, so these are invisible to it —
            including a dataset that a previous release produced and the current config
            no longer does, which is exactly the kind of disappearance a reader wants
            reported rather than silently omitted.
    """

    steps_without_datasets: list[str] = Field(default_factory=list)
    stages_without_config: list[str] = Field(default_factory=list)
    datasets_absent_from_both: list[str] = Field(default_factory=list)
    undeclared_in_buckets: list[str] = Field(default_factory=list)


def datasets_present(bucket: Bucket, prefix: str) -> set[str]:
    """List the release datasets actually present under a release root.

    Costs one full listing per namespace rather than one per dataset, so it is cheap
    relative to the per-dataset reads the comparison already performs.

    Args:
        bucket: The bucket to read.
        prefix: The release root prefix within it.

    Returns:
        Namespaced dataset names, e.g. `output/disease`, for every namespace directory
        that contains at least one object.
    """
    root = prefix.rstrip('/')
    found: set[str] = set()
    for namespace in ('output', 'view'):
        for blob in bucket.list_blobs(prefix=f'{root}/{namespace}/'):
            tail = blob.name[len(root) + 1 :]
            parts = tail.split('/')
            if len(parts) > 1 and parts[1]:
                found.add(f'{parts[0]}/{parts[1]}')
    return found


def collect_diffs(
    run_bucket: Bucket,
    run_prefix: str,
    reference_bucket: Bucket,
    reference_prefix: str,
    steps: list[str],
    stage_configs: dict[str, Any],
    read_footer: Callable[[str], Footer] | None = None,
) -> tuple[list[DatasetDiff], Skipped]:
    """Compare every release dataset the given steps declare.

    Args:
        run_bucket: The bucket holding the run.
        run_prefix: The run's root prefix within it.
        reference_bucket: The bucket holding the reference release.
        reference_prefix: The release's root prefix within it.
        steps: `unified_pipeline.yaml` step names to cover.
        stage_configs: Each stage's parsed config, keyed by stage name.
        read_footer: Passed through to `read_stats`.

    Returns:
        The diffs ordered by dataset, and a record of what was not covered. Both
        matter: a report that silently omits datasets reads as a clean comparison.
    """
    diffs: list[DatasetDiff] = []
    skipped = Skipped()
    seen: set[str] = set()

    for step in steps:
        config = stage_configs.get(identify(step).stage)
        if config is None:
            skipped.stages_without_config.append(step)
            continue
        destinations = destinations_for(step, config)
        if not destinations:
            skipped.steps_without_datasets.append(step)
            continue
        for dataset in destinations:
            if dataset in seen:
                continue
            seen.add(dataset)
            run = read_stats(run_bucket, f'{run_prefix.rstrip("/")}/{dataset}', read_footer)
            reference = read_stats(reference_bucket, f'{reference_prefix.rstrip("/")}/{dataset}', read_footer)
            if run is None and reference is None:
                skipped.datasets_absent_from_both.append(dataset)
                continue
            diffs.append(compare(dataset, run, reference))

    present = datasets_present(run_bucket, run_prefix) | datasets_present(reference_bucket, reference_prefix)
    skipped.undeclared_in_buckets = sorted(present - seen)

    diffs.sort(key=lambda d: d.dataset)
    return diffs, skipped


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
