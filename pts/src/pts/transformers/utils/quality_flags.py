"""Quality flag utilities for datasets processed with Polars."""

from collections.abc import Sequence
from enum import StrEnum
from typing import Any, NamedTuple, overload

import polars as pl
from loguru import logger

QC_COLUMN = 'qualityControls'


class ValidationResult[F: pl.DataFrame | pl.LazyFrame](NamedTuple):
    """Accepted and excluded partitions of a flagged dataset, eager or lazy as the dataset was."""

    valid: F
    invalid: F


def _quality_controls() -> pl.Expr:
    # An all-empty QC column can be typed List(Null), which set operations reject.
    return pl.col(QC_COLUMN).cast(pl.List(pl.String)).fill_null([])


def has_excluded_flag(invalid_flags: Sequence[str]) -> pl.Expr:
    """True where a row carries any of `invalid_flags`; null or empty QC lists carry none.

    `invalid_flags` are enum members or their values, such as the plain strings of a step's
    settings. An empty `invalid_flags` excludes nothing.
    """
    reasons = [str(flag) for flag in invalid_flags]
    return _quality_controls().list.set_intersection(reasons).list.len() > 0


@overload
def valid_rows(frame: pl.DataFrame, invalid_flags: Sequence[str]) -> ValidationResult[pl.DataFrame]: ...
@overload
def valid_rows(frame: pl.LazyFrame, invalid_flags: Sequence[str]) -> ValidationResult[pl.LazyFrame]: ...
def valid_rows(
    frame: pl.DataFrame | pl.LazyFrame, invalid_flags: Sequence[str]
) -> ValidationResult[pl.DataFrame | pl.LazyFrame]:
    """Split a dataset by `invalid_flags` (see `has_excluded_flag`), eager or lazy as it was given.

    With no `invalid_flags`, or no `qualityControls` column to carry flags, every row is valid and
    `frame` is returned as it is.
    """
    logger.info('Splitting dataset by {} excluded QC flags', len(invalid_flags))
    if not invalid_flags or QC_COLUMN not in frame.collect_schema():
        return ValidationResult(valid=frame, invalid=frame.clear())
    invalid = has_excluded_flag(invalid_flags)
    return ValidationResult(valid=frame.filter(~invalid), invalid=frame.filter(invalid))


@overload
def flag_rows(frame: pl.DataFrame, condition: pl.Expr, flag: StrEnum) -> pl.DataFrame: ...
@overload
def flag_rows(frame: pl.LazyFrame, condition: pl.Expr, flag: StrEnum) -> pl.LazyFrame: ...
def flag_rows(frame: pl.DataFrame | pl.LazyFrame, condition: pl.Expr, flag: StrEnum) -> pl.DataFrame | pl.LazyFrame:
    """Append `flag` to `qualityControls` where `condition` is true, keeping flags unique and sorted.

    Eager and lazy frames come back as the same kind. A missing `qualityControls` column is
    added, and null lists become empty on both branches, so every row stays countable as valid
    or invalid. Rows where `condition` is null are left unflagged. Null elements sort last, as
    Spark's `array_sort` does.
    """
    if QC_COLUMN not in frame.collect_schema():
        frame = frame.with_columns(pl.lit([], dtype=pl.List(pl.String)).alias(QC_COLUMN))
    qc = _quality_controls()
    appended = pl.concat_list(qc, pl.lit([flag.value])).list.unique().list.sort(nulls_last=True)
    return frame.with_columns(pl.when(condition).then(appended).otherwise(qc).alias(QC_COLUMN))


def flag_matching_ids(frame: pl.LazyFrame, key: str, ids: pl.LazyFrame, flag: StrEnum) -> pl.LazyFrame:
    """Append `flag` to the rows of `frame` whose `key` is among the `id` values of `ids`, keeping the row order."""
    matched = ids.select(pl.col('id').alias(key), pl.lit(True).alias('_matched')).unique()
    return (
        frame
        .join(matched, on=key, how='left', maintain_order='left')
        .pipe(flag_rows, pl.col('_matched').is_not_null(), flag)
        .drop('_matched')
    )


def resolve_flag_names(flags: type[StrEnum], value: Any) -> Any:
    """Map configured flag names (e.g. `UNRESOLVED_TARGET`) to members of `flags`.

    For pydantic `mode='before'` validators: flag values, members and other inputs pass
    through unchanged, so the field validation still reports unknown flags.
    """
    if isinstance(value, list):
        return [
            item if isinstance(item, flags) else flags.__members__.get(item, item) if isinstance(item, str) else item
            for item in value
        ]
    return value
