"""Comparing one run's datasets against a reference release.

Ported from the release-tested script at `~/Projects/compare_datasets`, which has run
against real releases. What carries over: reading a schema from a single parquet footer,
classifying columns as added, removed or retyped, walking `output/` and `view/` as
distinct namespaces, and treating a dataset present on only one side as a first-class
outcome rather than an error.

What is new here: row counts, which the original never had. It compared bytes only, so it
could not distinguish "this dataset grew 8%" from "this dataset lost forty thousand rows
and grew anyway". Counts come from every file's footer rather than one, so `O(files)`
metadata reads instead of `O(1)` — still no data scan.

NDJSON outputs, notably on the evidence path, have no footer at all. Those report bytes and
file counts only, and say so, rather than reporting a row count of zero that would read as
an empty dataset.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

Side = Literal['both', 'run_only', 'reference_only']
"""Whether a dataset appears on both sides of the comparison, or only one."""


class ColumnChange(BaseModel):
    """One column that differs between the run and the reference.

    Args:
        column: The column's name.
        kind: Whether it was added, removed, or kept its name and changed type.
        run_type: The type on the run side, absent when the column was removed.
        reference_type: The type on the reference side, absent when it was added.
    """

    column: str
    kind: Literal['added', 'removed', 'retyped']
    run_type: str | None = None
    reference_type: str | None = None


class DatasetDiff(BaseModel):
    """One dataset compared against its counterpart in the reference release.

    Args:
        dataset: The dataset's path relative to the release root, carrying its
            namespace, for example `output/disease` or `view/target`.
        side: Whether it exists on both sides or only one.
        run_rows: Row count on the run side, None when unavailable or absent.
        reference_rows: Row count on the reference side.
        run_bytes: Total bytes on the run side.
        reference_bytes: Total bytes on the reference side.
        run_files: File count on the run side, which catches repartitioning that
            leaves total size unchanged.
        reference_files: File count on the reference side.
        columns: Schema differences. Always reported, never thresholded.
        countable: False for formats with no footer, such as NDJSON. Describes the
            dataset's *format*, not whether either side's row count happens to be
            present — a `reference_only` dataset is legitimately `countable=True`
            with `run_rows=None`, because there is simply no run side to read a
            footer from. `is_material` does not branch on this field; it branches
            on `run_rows`/`reference_rows` being `None` directly, which is the
            right test regardless of why a count is missing.
    """

    dataset: str
    side: Side
    run_rows: int | None = None
    reference_rows: int | None = None
    run_bytes: int = 0
    reference_bytes: int = 0
    run_files: int = 0
    reference_files: int = 0
    columns: list[ColumnChange] = []
    countable: bool = True

    @property
    def row_delta(self) -> int | None:
        """Rows gained or lost against the reference.

        Returns:
            The difference, or None when either side has no count.
        """
        if self.run_rows is None or self.reference_rows is None:
            return None
        return self.run_rows - self.reference_rows

    @property
    def byte_delta(self) -> int:
        """Bytes gained or lost against the reference.

        Returns:
            The difference.
        """
        return self.run_bytes - self.reference_bytes


_BYTE_UNITS = ('B', 'kB', 'MB', 'GB', 'TB', 'PB')
"""Decimal units, deliberately, not the binary kiB/MiB set.

Google Cloud Storage sizes objects in powers of ten — what the console and `gcloud`
show — so a report that silently switched to powers of two would disagree with every
other view of the same bucket by 5% at MB scale and 10% at GB scale. That is the same
order as this comparison's own materiality threshold, which is exactly the wrong size
of discrepancy to introduce into a report about size changes.
"""


def human_bytes(value: int) -> str:
    """Render a byte count for a human reading a comment, not for a machine parsing one.

    Raw byte counts are what the comparison is computed on and what `--json` still
    emits; this is presentation only. Precision is deliberately dropped: the threshold
    that decides whether a change is worth reporting at all is fractional, so a change
    large enough to appear here is large enough to be visible at one decimal place.

    Args:
        value: A non-negative byte count.

    Returns:
        The count with a decimal unit suffix — `938 B`, `26.4 kB`, `85.0 MB`. Whole
        bytes below 1 kB, since a fractional byte is meaningless.
    """
    size = float(value)
    index = 0
    while size >= 1000 and index < len(_BYTE_UNITS) - 1:
        size /= 1000
        index += 1
    return f'{value} {_BYTE_UNITS[0]}' if index == 0 else f'{size:,.1f} {_BYTE_UNITS[index]}'


def compare_schemas(run: dict[str, str], reference: dict[str, str]) -> list[ColumnChange]:
    """Classify the differences between two schemas.

    Args:
        run: Column name to type, from the run.
        reference: Column name to type, from the reference.

    Returns:
        One entry per differing column, ordered added, removed, then retyped, and
        alphabetically within each. Empty when the schemas match.
    """
    added = [
        ColumnChange(column=c, kind='added', run_type=run[c]) for c in sorted(run.keys() - reference.keys())
    ]
    removed = [
        ColumnChange(column=c, kind='removed', reference_type=reference[c])
        for c in sorted(reference.keys() - run.keys())
    ]
    retyped = [
        ColumnChange(column=c, kind='retyped', run_type=run[c], reference_type=reference[c])
        for c in sorted(run.keys() & reference.keys())
        if run[c] != reference[c]
    ]
    return added + removed + retyped


def is_material(diff: DatasetDiff, threshold: float) -> bool:
    """Whether a diff is worth reporting.

    Schema changes and one-sided datasets are always material. Size and row moves are
    material only past the threshold, because run-to-run variation is normal and
    reporting all of it trains the reader to skim.

    Args:
        diff: The dataset comparison.
        threshold: Fractional change past which a count or size move is reported,
            for example 0.05 for five percent.

    Returns:
        True when the diff should appear in the report.
    """
    if diff.side != 'both' or diff.columns:
        return True

    for current, previous in ((diff.run_rows, diff.reference_rows), (diff.run_bytes, diff.reference_bytes)):
        if current is None or previous is None:
            continue
        if previous == 0:
            if current != 0:
                return True
            continue
        if abs(current - previous) / previous > threshold:
            return True
    return False


def schema_of(footer: Any) -> dict[str, str]:
    """Reduce a parquet footer to a column-name-to-type mapping.

    Args:
        footer: A `pyarrow` schema.

    Returns:
        Column name to stringified type.
    """
    return {field.name: str(field.type) for field in footer}
