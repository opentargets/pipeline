"""Polars equivalents of spark's frame operations."""

from __future__ import annotations

from typing import overload

import polars as pl


@overload
def spark_distinct(frame: pl.DataFrame) -> pl.DataFrame: ...
@overload
def spark_distinct(frame: pl.LazyFrame) -> pl.LazyFrame: ...
def spark_distinct(frame: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
    """Spark's `distinct` (-0.0 as 0.0 in float columns), keeping the first of each duplicate in order."""
    zeros_normalised = [
        pl.when(pl.col(name) == 0.0).then(pl.lit(0.0, dtype=dtype)).otherwise(pl.col(name)).alias(name)
        for name, dtype in frame.collect_schema().items()
        if dtype.is_float()
    ]
    return frame.with_columns(zeros_normalised).unique(keep='first', maintain_order=True)
