"""Polars equivalents of spark's maths functions and its p-value exponent/mantissa split."""

import functools
import math
import struct
from typing import overload

import numpy as np
import polars as pl
from loguru import logger

from pts.transformers.evidence.utils.expressions import divide
from pts.transformers.utils.spark_strings import java_double_to_string

_INT_MAX = 2**31 - 1


_TWO54 = 1.80143985094819840000e16
_LN2_HI, _LN2_LO = 6.93147180369123816490e-01, 1.90821492927058770002e-10
_LG1, _LG2, _LG3 = 6.666666666666735130e-01, 3.999999999940941908e-01, 2.857142874366239149e-01
_LG4, _LG5, _LG6 = 2.222219843214978396e-01, 1.818357216161805012e-01, 1.531383769920937332e-01
_LG7 = 1.479819860511658591e-01
_IVLN10 = 4.34294481903251816668e-01
_LOG10_2HI, _LOG10_2LO = 3.01029995663611771306e-01, 3.69423907715893078616e-13


def _words(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bits = x.view(np.int64)
    return bits >> 32, bits & 0xFFFFFFFF


def _join_words(hi: np.ndarray, lo: np.ndarray) -> np.ndarray:
    return (((hi & 0xFFFFFFFF) << 32) | lo).view(np.float64)


def _fdlibm_log(x: np.ndarray) -> np.ndarray:
    """The fdlibm `e_log.c` for normal `x` in `[1, 2)` or `[0.5, 1)`."""
    hx, lx = _words(x)
    k = (hx >> 20) - 1023
    hx = hx & 0x000FFFFF
    i = (hx + 0x95F64) & 0x100000
    x = _join_words(hx | (i ^ 0x3FF00000), lx)
    k = k + (i >> 20)
    f = x - 1.0
    dk = k.astype(np.float64)
    small = (0x000FFFFF & (2 + hx)) < 3
    r_small = f * f * (0.5 - 0.33333333333333333 * f)
    small_result = np.where(
        f == 0.0,
        np.where(k == 0, 0.0, dk * _LN2_HI + dk * _LN2_LO),
        np.where(k == 0, f - r_small, dk * _LN2_HI - ((r_small - dk * _LN2_LO) - f)),
    )
    s = f / (2.0 + f)
    z = s * s
    w = z * z
    t1 = w * (_LG2 + w * (_LG4 + w * _LG6))
    t2 = z * (_LG1 + w * (_LG3 + w * (_LG5 + w * _LG7)))
    r = t2 + t1
    hfsq = 0.5 * f * f
    wide = ((hx - 0x6147A) | (0x6B851 - hx)) > 0
    wide_result = np.where(
        k == 0,
        f - (hfsq - s * (hfsq + r)),
        dk * _LN2_HI - ((hfsq - (s * (hfsq + r) + dk * _LN2_LO)) - f),
    )
    narrow_result = np.where(k == 0, f - s * (f - r), dk * _LN2_HI - ((s * (f - r) - dk * _LN2_LO) - f))
    return np.where(small, small_result, np.where(wide, wide_result, narrow_result))


def _fdlibm_log10(x: np.ndarray) -> np.ndarray:
    """The fdlibm `e_log10.c` (java's `StrictMath.log10`) for finite `x > 0`."""
    subnormal = x < 2.0**-1022
    x = np.where(subnormal, x * _TWO54, x)
    hx, lx = _words(x)
    k = np.where(subnormal, -54, 0) + (hx >> 20) - 1023
    i = (k < 0).astype(np.int64)
    y = (k + i).astype(np.float64)
    normalised = _join_words((hx & 0x000FFFFF) | ((0x3FF - i) << 20), lx)
    z = y * _LOG10_2LO + _IVLN10 * _fdlibm_log(normalised)
    return z + y * _LOG10_2HI


def _strict_log10(series: pl.Series) -> pl.Series:
    x = series.cast(pl.Float64).to_numpy()
    values = np.full(len(x), np.nan)
    finite = (x > 0) & np.isfinite(x)
    values[finite] = _fdlibm_log10(x[finite])
    values[x == np.inf] = np.inf
    out = pl.Series(series.name, values, dtype=pl.Float64, nan_to_null=False)
    nulls = np.flatnonzero(series.is_null().to_numpy() | (x <= 0) | (x == -np.inf))
    return out.scatter(nulls, None) if len(nulls) else out


def spark_log10(expr: pl.Expr) -> pl.Expr:
    """Spark's `log10` (fdlibm, as java's `StrictMath`): null for a value `<= 0`, NaN for NaN."""
    return expr.map_batches(_strict_log10, return_dtype=pl.Float64, is_elementwise=True)


def _spark_to_int(expr: pl.Expr) -> pl.Expr:
    """Spark's non-ANSI `CAST(double AS INT)`: truncated, NaN 0, infinities clamped to the Int32 range."""
    return (
        pl
        .when(expr.is_nan())
        .then(pl.lit(0, dtype=pl.Int32))
        .when(expr == math.inf)
        .then(pl.lit(_INT_MAX, dtype=pl.Int32))
        .when(expr == -math.inf)
        .then(pl.lit(-_INT_MAX - 1, dtype=pl.Int32))
        .otherwise(expr.cast(pl.Int32, strict=False))
    )


def pvalue_exponent(p: pl.Expr) -> pl.Expr:
    """`CAST(log10(p) AS INT) - 1` as spark computes it, Int32."""
    return _spark_to_int(spark_log10(p)) - 1


def _bits(x: float) -> int:
    return struct.unpack('<q', struct.pack('<d', x))[0]


def _hi(x: float) -> int:
    return _bits(x) >> 32


def _lo(x: float) -> int:
    return _bits(x) & 0xFFFFFFFF


def _from_words(hi: int, lo: int) -> float:
    return struct.unpack('<d', struct.pack('<Q', ((hi & 0xFFFFFFFF) << 32) | lo))[0]


def _clear_lo(x: float) -> float:
    return _from_words(_hi(x), 0)


_BP = (1.0, 1.5)
_DP_H = (0.0, 5.84962487220764160156e-01)
_DP_L = (0.0, 1.35003920212974897128e-08)
_L1, _L2, _L3 = 5.99999999999994648725e-01, 4.28571428578550184252e-01, 3.33333329818377432918e-01
_L4, _L5, _L6 = 2.72728123808534006489e-01, 2.30660745775561754067e-01, 2.06975017800338417784e-01
_P1, _P2, _P3 = 1.66666666666666019037e-01, -2.77777777770155933842e-03, 6.61375632143793436117e-05
_P4, _P5 = -1.65339022054652515390e-06, 4.13813679705723846039e-08
_POW_LG2, _POW_LG2_H, _POW_LG2_L = 6.93147180559945286227e-01, 6.93147182464599609375e-01, -1.90465429995776804525e-09
_OVT = 8.0085662595372944372e-17
_CP, _CP_H, _CP_L = 9.61796693925975554329e-01, 9.61796700954437255859e-01, -7.02846165095275826516e-09
_TWO53 = 9007199254740992.0


@functools.cache
def strict_pow(x: float, y: float) -> float:
    """Java's `StrictMath.pow` (fdlibm `e_pow.c`), parity-tested for `x = 10` and an integer `y`."""
    if y == 0:
        return 1.0
    if y == 1:
        return x
    if y == -1:
        return 1.0 / x
    if y == 2:
        return x * x
    if y == 0.5:
        return math.sqrt(x)
    ix = _hi(x) & 0x7FFFFFFF
    iy = _hi(y) & 0x7FFFFFFF
    if iy > 0x41E00000:
        return math.inf if (x > 1) == (y > 0) else 0.0
    ax = x
    n = 0
    if ix < 0x00100000:
        ax *= _TWO53
        n -= 53
        ix = _hi(ax)
    n += (ix >> 20) - 0x3FF
    j = ix & 0x000FFFFF
    ix = j | 0x3FF00000
    if j <= 0x3988E:
        k = 0
    elif j < 0xBB67A:
        k = 1
    else:
        k = 0
        n += 1
        ix -= 0x00100000
    ax = _from_words(ix, _lo(ax))

    u = ax - _BP[k]
    v = 1.0 / (ax + _BP[k])
    ss = u * v
    s_h = _clear_lo(ss)
    t_h = _from_words(((ix >> 1) | 0x20000000) + 0x00080000 + (k << 18), 0)
    t_l = ax - (t_h - _BP[k])
    s_l = v * ((u - s_h * t_h) - s_h * t_l)
    s2 = ss * ss
    r = s2 * s2 * (_L1 + s2 * (_L2 + s2 * (_L3 + s2 * (_L4 + s2 * (_L5 + s2 * _L6)))))
    r += s_l * (s_h + ss)
    s2 = s_h * s_h
    t_h = _clear_lo(3.0 + s2 + r)
    t_l = r - ((t_h - 3.0) - s2)
    u = s_h * t_h
    v = s_l * t_h + t_l * ss
    p_h = _clear_lo(u + v)
    p_l = v - (p_h - u)
    z_h = _CP_H * p_h
    z_l = _CP_L * p_h + p_l * _CP + _DP_L[k]
    t = float(n)
    t1 = _clear_lo(((z_h + z_l) + _DP_H[k]) + t)
    t2 = z_l - (((t1 - t) - _DP_H[k]) - z_h)

    y1 = _clear_lo(y)
    p_l = (y - y1) * t1 + y * t2
    p_h = y1 * t1
    z = p_l + p_h
    j = _hi(z)
    i = _lo(z)
    if j >= 0x40900000:
        if ((j - 0x40900000) | i) != 0 or p_l + _OVT > z - p_h:
            return math.inf
    elif (j & 0x7FFFFFFF) >= 0x4090CC00:
        if ((j & 0xFFFFFFFF) - 0xC090CC00) | i != 0 or p_l <= z - p_h:
            return 0.0
    i = j & 0x7FFFFFFF
    k = (i >> 20) - 0x3FF
    n = 0
    if i > 0x3FE00000:
        n = j + (0x00100000 >> (k + 1))
        k = ((n & 0x7FFFFFFF) >> 20) - 0x3FF
        t = _from_words(n & ~(0x000FFFFF >> k), 0)
        n = ((n & 0x000FFFFF) | 0x00100000) >> (20 - k)
        if j < 0:
            n = -n
        p_h -= t
    t = _clear_lo(p_l + p_h)
    u = t * _POW_LG2_H
    v = (p_l - (t - p_h)) * _POW_LG2 + t * _POW_LG2_L
    z = u + v
    w = v - (z - u)
    t = z * z
    t1 = z - t * (_P1 + t * (_P2 + t * (_P3 + t * (_P4 + t * _P5))))
    r = (z * t1) / (t1 - 2.0) - (w + z * w)
    z = 1.0 - (r - z)
    j = _hi(z) + (n << 20)
    if (j >> 20) <= 0:
        return math.ldexp(z, n)
    return _from_words(j, _lo(z))


def _divide_by_power_of_ten(columns: pl.Series) -> pl.Series:
    """Spark's `value / pow(10, exponent)`: IEEE division, null for a zero divisor."""
    value = columns.struct.field('value')
    exponent = columns.struct.field('exponent')
    powers = {k: strict_pow(10.0, float(k)) for k in exponent.drop_nulls().unique().to_list()}
    divisor = exponent.replace_strict(powers, default=None, return_dtype=pl.Float64).to_numpy()
    with np.errstate(all='ignore'):
        values = value.cast(pl.Float64).to_numpy() / divisor
    out = pl.Series(columns.name, values, dtype=pl.Float64, nan_to_null=False)
    nulls = np.flatnonzero((value.is_null() | exponent.is_null()).to_numpy() | (divisor == 0))
    return out.scatter(nulls, None) if len(nulls) else out


def pvalue_mantissa(p: pl.Expr) -> pl.Expr:
    """`round(p / pow(10, exponent), 3)` as spark computes it, with `pvalue_exponent`: 10.0 at a power of ten."""
    quotient = pl.struct(p.alias('value'), pvalue_exponent(p).alias('exponent')).map_batches(
        _divide_by_power_of_ten, return_dtype=pl.Float64, is_elementwise=True
    )
    return round_half_up(quotient, 3)


def round_half_up(expr: pl.Expr, scale: int) -> pl.Expr:
    """Spark's `round(double, scale)`: HALF_UP on java's `Double.toString`, not on the binary value.

    NaN and infinities are kept; a result of zero is `0.0`, never `-0.0`.
    """
    text = java_double_to_string(expr).str.strip_chars_start('-')
    split = text.str.split_exact('E', 1)
    mantissa = split.struct.field('field_0')
    digits = mantissa.str.replace('.', '', literal=True)
    significant = digits.str.strip_chars_start('0')
    exponent = (
        mantissa.str.find('.', literal=True).cast(pl.Int32)
        - 1
        - (digits.str.len_chars() - significant.str.len_chars()).cast(pl.Int32)
        + split.struct.field('field_1').cast(pl.Int32, strict=False).fill_null(0)
    )
    kept_length = exponent + 1 + scale
    kept = significant.str.slice(0, kept_length.clip(0)).replace('', '0').cast(pl.Int64, strict=False) + (
        significant.str.slice(kept_length.clip(0), 1) >= '5'
    ).cast(pl.Int64)
    rounded = pl.concat_str(
        pl.when(expr < 0).then(pl.lit('-')).otherwise(pl.lit('')), kept.cast(pl.String), pl.lit(f'e{-scale}')
    ).cast(pl.Float64)
    return (
        pl
        .when(expr.is_nan() | expr.is_infinite())
        .then(expr)
        .when((significant == '') | (kept_length < 0))
        .then(pl.lit(0.0))
        .when(kept_length >= significant.str.len_chars().cast(pl.Int32))
        .then(expr)
        .when(kept == 0)
        .then(pl.lit(0.0))
        .otherwise(rounded)
    )


def normalise_mantissa(mantissa: pl.Expr, exponent: pl.Expr) -> tuple[pl.Expr, pl.Expr]:
    """A mantissa `>= 10` divided by 10 and its exponent raised by one; anything else, NaN included, kept."""
    shift = (mantissa >= 10) & ~mantissa.is_nan()
    return (
        pl.when(shift).then(divide(mantissa, 10.0)).otherwise(mantissa),
        pl.when(shift).then(exponent + 1).otherwise(exponent),
    )


@overload
def substitute_zero_pvalues(frame: pl.DataFrame, column: str, label: str | None = None) -> pl.DataFrame: ...
@overload
def substitute_zero_pvalues(frame: pl.LazyFrame, column: str, label: str | None = None) -> pl.LazyFrame: ...
def substitute_zero_pvalues(
    frame: pl.DataFrame | pl.LazyFrame, column: str, label: str | None = None
) -> pl.DataFrame | pl.LazyFrame:
    """Replace a p-value of 0.0 with the frame's smallest p-value above 0, null if there is none.

    The count is logged under `label`, the column name by default.
    """
    pvalue = pl.col(column)
    zeros, minimum = (
        frame.lazy().select((pvalue == 0.0).sum(), pvalue.filter(pvalue > 0.0).min().alias('minimum')).collect().row(0)
    )
    if not zeros:
        return frame
    logger.warning(
        f'There are {zeros} {label or column} evidence with a p-value of 0.0; they take the minimum, {minimum}.'
    )
    dtype = frame.collect_schema()[column]
    return frame.with_columns(pl.when(pvalue == 0.0).then(pl.lit(minimum, dtype=dtype)).otherwise(pvalue).alias(column))
