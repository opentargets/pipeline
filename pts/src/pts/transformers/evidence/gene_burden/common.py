"""Shared pieces of the gene burden sub-source parsers."""

from __future__ import annotations

from typing import overload

import polars as pl

from pts.schemas.gene_burden import GENE_BURDEN_UNION

DATASOURCE_ID = 'gene_burden'
DATATYPE_ID = 'genetic_association'


@overload
def to_union(frame: pl.DataFrame) -> pl.DataFrame: ...
@overload
def to_union(frame: pl.LazyFrame) -> pl.LazyFrame: ...
def to_union(frame: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
    """The frame in `GENE_BURDEN_UNION`: its columns in order and cast, the missing ones null."""
    names = frame.collect_schema().names()
    return frame.select(
        pl.col(name).cast(dtype) if name in names else pl.lit(None, dtype=dtype).alias(name)
        for name, dtype in GENE_BURDEN_UNION.items()
    )


@overload
def spark_distinct(frame: pl.DataFrame) -> pl.DataFrame: ...
@overload
def spark_distinct(frame: pl.LazyFrame) -> pl.LazyFrame: ...
def spark_distinct(frame: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
    """Spark's `distinct`: -0.0 comes out as 0.0 in float columns, and the first of each duplicate is kept in order."""
    zeros_normalised = [
        pl.when(pl.col(name) == 0.0).then(pl.lit(0.0, dtype=dtype)).otherwise(pl.col(name)).alias(name)
        for name, dtype in frame.collect_schema().items()
        if dtype.is_float()
    ]
    return frame.with_columns(zeros_normalised).unique(keep='first', maintain_order=True)
