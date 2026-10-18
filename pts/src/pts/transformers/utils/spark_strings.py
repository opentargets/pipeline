"""Polars equivalents of spark's string functions, where the two disagree."""

import polars as pl


def spark_trim(expr: pl.Expr) -> pl.Expr:
    """Spark's `trim`: strip the space character from both ends, not whitespace in general."""
    return expr.str.strip_chars(' ')


def _decimal_digits(text: pl.Expr, marker: str) -> tuple[pl.Expr, pl.Expr]:
    """Unsigned `<mantissa>[<marker><n>]` text as its significant digits and the decimal exponent of the first."""
    split = text.str.split_exact(marker, 1)
    mantissa = split.struct.field('field_0')
    digits = mantissa.str.replace('.', '', literal=True)
    significant = digits.str.strip_chars_start('0')
    exponent = (
        mantissa.str.find('.', literal=True).fill_null(mantissa.str.len_chars()).cast(pl.Int32)
        - 1
        - (digits.str.len_chars() - significant.str.len_chars()).cast(pl.Int32)
        + split.struct.field('field_1').cast(pl.Int32, strict=False).fill_null(0)
    )
    return significant, exponent


def java_double_to_string(expr: pl.Expr) -> pl.Expr:
    """Spark's `CAST(double AS STRING)`: java's `Double.toString`, null-preserving.

    Plain notation inside `1e-3 <= |x| < 1e7`, otherwise `d.dddE<n>`; a whole number keeps `.0`;
    `NaN`, `Infinity`, `-Infinity`. Digits are polars' shortest round-trip ones, except integral
    doubles in `[2**53, 2**63)`, which java writes from their long value rounded half-up by 0-2
    digits. Where java 17 is otherwise not shortest (some subnormals, `|x| >= 2**63`) this is.
    """
    rendered = expr.cast(pl.String)
    magnitude = expr.abs()
    scientific_range = ((magnitude < 1e-3) | (magnitude >= 1e7)) & (magnitude > 0)
    significant, exponent = _decimal_digits(pl.when(scientific_range).then(rendered.str.strip_chars_start('-')), 'e')
    is_long = (magnitude >= 2.0**53) & (magnitude < 2.0**63)
    dropped = pl.when(magnitude >= 2.0**61).then(2).when(magnitude >= 2.0**58).then(1).otherwise(0)
    scale = pl.lit(10, dtype=pl.Int64).pow(dropped)
    whole = pl.when(is_long).then(magnitude.cast(pl.Int64, strict=False))
    long_digits = (whole // scale + (whole % scale * 2 >= scale).cast(pl.Int64)).cast(pl.String)
    exponent = pl.when(is_long).then(long_digits.str.len_chars().cast(pl.Int32) - 1 + dropped).otherwise(exponent)
    significant = pl.coalesce(long_digits, significant).str.strip_chars_end('0').str.pad_end(2, '0')
    scientific = pl.concat_str(
        pl.when(expr < 0).then(pl.lit('-')).otherwise(pl.lit('')),
        significant.str.slice(0, 1),
        pl.lit('.'),
        significant.str.slice(1),
        pl.lit('E'),
        exponent.cast(pl.String),
    )
    return (
        pl.when(expr.is_nan())
        .then(pl.lit('NaN'))
        .when(expr.is_infinite())
        .then(pl.when(expr > 0).then(pl.lit('Infinity')).otherwise(pl.lit('-Infinity')))
        .otherwise(pl.coalesce(scientific, rendered))
    )


def regexp_extract(value: pl.Expr, pattern: str, group: int) -> pl.Expr:
    """Spark's `regexp_extract`: the group's match, empty if there is none, null for a null value."""
    return pl.when(value.is_not_null()).then(value.str.extract(pattern, group).fill_null(''))
