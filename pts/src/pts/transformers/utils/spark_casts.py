"""Polars equivalents of spark's non-ANSI casts."""

import polars as pl

SPARK_CAST_TRIM = ''.join(chr(c) for c in range(0x21))
"""Characters spark strips before casting a string to a number."""


def cast_double(column: str) -> pl.Expr:
    """Cast a text column to double as spark's non-ANSI cast does, giving null on bad text."""
    value = pl.col(column).str.strip_chars(SPARK_CAST_TRIM)
    number = value.str.replace(r'^([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)[dDfF]$', '$1')
    signed_nan = value.str.contains(r'^[+-][nN][aA][nN]$') & ~value.str.contains(r'^[+-]NaN$')
    return pl.when(~signed_nan).then(number.cast(pl.Float64, strict=False)).alias(column)
