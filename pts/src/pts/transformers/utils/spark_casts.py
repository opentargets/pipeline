"""Polars equivalents of spark's non-ANSI casts."""

import polars as pl

SPARK_CAST_TRIM = ''.join(chr(c) for c in range(0x21))
"""Characters spark strips before casting a string to a float or double."""

SPARK_INTEGER_CAST_TRIM = SPARK_CAST_TRIM + '\x7f'
"""Characters spark strips before casting a string to an integer type: those above, and DEL."""


def to_long(col: pl.Expr) -> pl.Expr:
    """Cast text to Int64: surrounding whitespace ignored, a fraction truncated, anything else null."""
    trimmed = col.str.strip_chars(SPARK_INTEGER_CAST_TRIM)
    whole = trimmed.str.extract(r'^([+-]?[0-9]*)(?:\.[0-9]*)?$', 1)
    return (
        pl
        .when(whole.str.contains('[0-9]'))
        .then(whole)
        .when(whole.is_not_null() & trimmed.str.contains('.', literal=True))
        .then(pl.lit('0'))
        .cast(pl.Int64, strict=False)
    )


def to_int(col: pl.Expr) -> pl.Expr:
    """`to_long`, null outside the Int32 range."""
    return to_long(col).cast(pl.Int32, strict=False)


def _float_text(col: pl.Expr) -> pl.Expr:
    """Text as spark's float and double casts read it: trimmed, a `d`/`f` suffix dropped, signed `nan` null."""
    value = col.str.strip_chars(SPARK_CAST_TRIM)
    # java's decimal syntax, as `spark_csv._JAVA_DECIMAL` reads it
    number = value.str.replace(r'^([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)[dDfF]$', '$1')
    signed_nan = value.str.contains(r'^[+-][nN][aA][nN]$') & ~value.str.contains(r'^[+-]NaN$')
    return pl.when(~signed_nan).then(number)


def to_float(col: pl.Expr) -> pl.Expr:
    """Cast text to Float32 as spark's non-ANSI cast does, giving null on bad text."""
    return _float_text(col).cast(pl.Float32, strict=False)


def cast_double(column: str) -> pl.Expr:
    """Cast a text column to double as spark's non-ANSI cast does, giving null on bad text."""
    return _float_text(pl.col(column)).cast(pl.Float64, strict=False).alias(column)
