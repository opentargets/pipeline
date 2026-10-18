"""Polars equivalents of spark's string functions, where the two disagree."""

import functools
import math
from fractions import Fraction

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


def _java_17_rendering(value: float) -> str:
    """JDK 17's `Double.toString` of a positive finite double, from `FloatingDecimal.dtoa`'s digit loop.

    The exact value's digits, rounded half-even, at the first length where rounding down or up lands
    strictly within `half_gap` of the value: half the gap to the next double, a quarter for a power
    of two (subnormal ones included). Scientific notation takes two digits at least, unless dtoa's
    decimal exponent estimate was one too high. A carry out keeps the length, so trailing zeros stay.
    """
    mantissa, exponent = math.frexp(value)
    exact = Fraction(value)
    gap = Fraction(2) ** max(exponent - 53, -1074)
    half_gap = gap / 4 if mantissa == 0.5 else gap / 2
    estimate = math.floor((2 * mantissa - 1.5) * 0.289529654 + 0.176091259 + (exponent - 1) * 0.301029995663981)
    decimal = estimate - (Fraction(10) ** estimate > exact)
    plain = -3 <= decimal < 7
    length = 1 if plain or decimal < estimate else 2
    while True:
        unit = Fraction(10) ** (decimal + 1 - length)
        kept, rest = divmod(exact, unit)
        if rest < half_gap or unit - rest < half_gap:
            break
        length += 1
    kept += rest * 2 > unit or (rest * 2 == unit and kept % 2 == 1)
    digits = str(kept)
    decimal += len(digits) - length
    digits = digits[:length]
    if not plain:
        return f'{digits[0]}.{digits[1:] or "0"}E{decimal}'
    if decimal < 0:
        return f'0.{"0" * (-decimal - 1)}{digits}'
    return f'{digits[: decimal + 1].ljust(decimal + 1, "0")}.{digits[decimal + 1 :] or "0"}'


@functools.cache
def _java_17_irregular() -> dict[float, str]:
    """`_java_17_rendering` of every power of two outside `[2**53, 2**63)` and every subnormal below 32 ulps."""
    powers = [math.ldexp(1.0, e) for e in range(-1074, 1024) if not 53 <= e < 63]
    subnormals = [math.ldexp(float(n), -1074) for n in range(1, 32)]
    return {v: _java_17_rendering(v) for v in powers + subnormals}


def java_double_to_string(expr: pl.Expr) -> pl.Expr:
    """Spark's `CAST(double AS STRING)`: JDK 17's `Double.toString`, null-preserving.

    Plain notation inside `1e-3 <= |x| < 1e7`, otherwise `d.dddE<n>`; a whole number keeps `.0`;
    `NaN`, `Infinity`, `-Infinity`. Digits are polars' shortest round-trip ones, except integral
    doubles in `[2**53, 2**63)`, which java writes from their long value rounded half-up by 0-2
    digits, and powers of two and subnormals below 32 ulps, where java 17 is not shortest and
    `_java_17_rendering` reproduces it. That is java 17 for every double below `2**63` and every
    power of two. Above `2**63` java 17 is also not shortest for some other doubles, and this is:
    where a long overflows in its digit loop, or a shortest candidate lies exactly half a gap away.
    """
    rendered = expr.cast(pl.String)
    magnitude = expr.abs()
    sign = pl.when(expr < 0).then(pl.lit('-')).otherwise(pl.lit(''))
    irregular = _java_17_irregular()
    looked_up = pl.concat_str(
        sign, magnitude.replace_strict(list(irregular), list(irregular.values()), default=None, return_dtype=pl.String)
    )
    scientific_range = ((magnitude < 1e-3) | (magnitude >= 1e7)) & (magnitude > 0)
    significant, exponent = _decimal_digits(pl.when(scientific_range).then(rendered.str.strip_chars_start('-')), 'e')
    is_long = (magnitude >= 2.0**53) & (magnitude < 2.0**63)
    dropped = pl.when(magnitude >= 2.0**61).then(2).when(magnitude >= 2.0**58).then(1).otherwise(0)
    scale = pl.lit(10, dtype=pl.Int64).pow(dropped)
    whole = pl.when(is_long).then(magnitude.cast(pl.Int64, strict=False))
    long_digits = (whole // scale + (whole % scale * 2 >= scale).cast(pl.Int64)).cast(pl.String)
    exponent = pl.coalesce(long_digits.str.len_chars().cast(pl.Int32) - 1 + dropped, exponent)
    significant = pl.coalesce(long_digits, significant).str.strip_chars_end('0').str.pad_end(2, '0')
    scientific = pl.concat_str(
        sign,
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
        .otherwise(pl.coalesce(looked_up, scientific, rendered))
    )


def regexp_extract(value: pl.Expr, pattern: str, group: int) -> pl.Expr:
    """Spark's `regexp_extract`: the group's match, empty if there is none, null for a null value."""
    return pl.when(value.is_not_null()).then(value.str.extract(pattern, group).fill_null(''))
