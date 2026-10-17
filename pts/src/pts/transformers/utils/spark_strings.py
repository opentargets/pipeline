"""Polars equivalents of spark's string functions, where the two disagree."""

import polars as pl


def spark_trim(expr: pl.Expr) -> pl.Expr:
    """Spark's `trim`: strip the space character from both ends, not whitespace in general."""
    return expr.str.strip_chars(' ')


def regexp_extract(value: pl.Expr, pattern: str, group: int) -> pl.Expr:
    """Spark's `regexp_extract`: the group's match, empty if there is none, null for a null value."""
    return pl.when(value.is_not_null()).then(value.str.extract(pattern, group).fill_null(''))
