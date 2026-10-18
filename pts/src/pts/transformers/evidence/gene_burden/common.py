"""Shared pieces of the gene burden sub-source parsers."""

from __future__ import annotations

from typing import overload

import polars as pl

from pts.schemas.gene_burden import GENE_BURDEN_PARSED

DATASOURCE_ID = 'gene_burden'
DATATYPE_ID = 'genetic_association'


def lookup(expr: pl.Expr, mapping: dict[str, str]) -> pl.Expr:
    """Spark's `create_map(...)[expr]`: the mapped value, null for a missing or null key."""
    return expr.replace_strict(mapping, default=None, return_dtype=pl.String)


@overload
def to_parsed(frame: pl.DataFrame) -> pl.DataFrame: ...
@overload
def to_parsed(frame: pl.LazyFrame) -> pl.LazyFrame: ...
def to_parsed(frame: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
    """The frame in `GENE_BURDEN_PARSED`: its columns in order and cast, the missing ones null."""
    names = frame.collect_schema().names()
    return frame.select(
        pl.col(name).cast(dtype) if name in names else pl.lit(None, dtype=dtype).alias(name)
        for name, dtype in GENE_BURDEN_PARSED.items()
    )
