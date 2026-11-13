"""Shared list column helpers for polars transformers."""

import polars as pl


def explode(lf: pl.LazyFrame, column: str) -> pl.LazyFrame:
    """Explode a list column, dropping rows whose list is null or empty."""
    return lf.filter(pl.col(column).list.len() > 0).explode(column)


def collect_set(column: str) -> pl.Expr:
    """Distinct non-null values of a column, sorted."""
    return pl.col(column).drop_nulls().unique().sort()


def tag(expr: pl.Expr, field: str, source: str) -> pl.Expr:
    """Wrap each element of a string list as a `{field, source}` struct; null stays null."""
    return expr.list.eval(pl.struct(pl.element().alias(field), pl.lit(source).alias('source')))
