"""Quality flag utilities for datasets processed with Polars."""

from collections.abc import Sequence
from enum import StrEnum
from typing import Any, NamedTuple

import polars as pl
from loguru import logger


class ValidationResult(NamedTuple):
    """Accepted and excluded partitions of a flagged dataset."""

    valid: pl.LazyFrame
    invalid: pl.LazyFrame


def _quality_controls() -> pl.Expr:
    # An all-empty QC column can be typed List(Null), which set operations reject.
    return pl.col('qualityControls').cast(pl.List(pl.String)).fill_null([])


def has_excluded_flag(invalid_flags: Sequence[StrEnum]) -> pl.Expr:
    """True where a row carries any of `invalid_flags`; null or empty QC lists carry none."""
    reasons = [flag.value for flag in invalid_flags]
    return _quality_controls().list.set_intersection(reasons).list.len() > 0


def valid_rows(frame: pl.LazyFrame, invalid_flags: Sequence[StrEnum]) -> ValidationResult:
    """Split a dataset by configured QC enum values, treating null QC lists as empty."""
    logger.info('Splitting dataset by {} excluded QC flags', len(invalid_flags))
    invalid = has_excluded_flag(invalid_flags)
    return ValidationResult(valid=frame.filter(~invalid), invalid=frame.filter(invalid))


def flag_rows(frame: pl.LazyFrame, condition: pl.Expr, flag: StrEnum) -> pl.LazyFrame:
    """Append `flag` to `qualityControls` where `condition` is true, keeping flags unique and sorted.

    A missing `qualityControls` column is added, and null lists become empty. Rows where
    `condition` is null are left unflagged.
    """
    if 'qualityControls' not in frame.collect_schema():
        frame = frame.with_columns(pl.lit([], dtype=pl.List(pl.String)).alias('qualityControls'))
    qc = _quality_controls()
    return frame.with_columns(
        pl
        .when(condition)
        .then(pl.concat_list(qc, pl.lit([flag.value])).list.unique().list.sort())
        .otherwise(qc)
        .alias('qualityControls')
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


def update_quality_flag(
    df: pl.DataFrame,
    flag_condition: pl.Expr,
    flag_text: str,
) -> pl.DataFrame:
    """Update the qualityControls column in a Polars DataFrame based on a flag condition.

    Args:
        df: The input Polars DataFrame
        flag_condition: A Polars expression that evaluates to a boolean
        flag_text: The text to add to the qualityControls list if the condition is met

    Returns:
        A new Polars DataFrame with updated qualityControls column
    """
    # First ensure qualityControls column exists
    if 'qualityControls' not in df.columns:
        df = df.with_columns(pl.lit(None, dtype=pl.List(pl.String)).alias('qualityControls'))

    return df.with_columns(
        pl
        .when(flag_condition)
        .then(
            pl
            .concat_list([
                pl.col('qualityControls').fill_null([]),
                pl.lit(flag_text).implode(),
            ])
            .list.unique()
            .list.sort()
        )
        .otherwise(pl.col('qualityControls'))
        .alias('qualityControls')
    )
