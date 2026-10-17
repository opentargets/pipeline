"""Tests for the polars equivalents of spark's p-value maths."""

import math
import random

import polars as pl
import pyspark.sql.functions as f
import pytest
from loguru import logger

from pts.pyspark.gene_burden import _substitute_zero_pvalues
from pts.transformers.utils.spark_math import (
    normalise_mantissa,
    pvalue_exponent,
    pvalue_mantissa,
    round_half_up,
    spark_log10,
    strict_pow,
    substitute_zero_pvalues,
)

FOCUS_PVALUES = [0.0, -0.0, 1e-8, 5e-320, math.nan, 1.0, 1.6688e-308, 5.498e-320, 4.9e-324, math.inf, -1e-5, None]


def _pvalues() -> list[float | None]:
    rng = random.Random(20261017)
    values: list[float | None] = list(FOCUS_PVALUES)
    for k in range(-323, 1):
        power = float(f'1e{k}')
        values += [power, math.nextafter(power, 0.0), math.nextafter(power, 1.0)]
    values += [10 ** rng.uniform(-320, 0) for _ in range(5000)]
    values += [float(f'{rng.randint(1000, 9999)}5e{rng.randint(-320, -4)}') for _ in range(5000)]
    values += [1.2345e-9, 9.9995e-9, 9.9994999e-9, 1.0005e-3]
    return values


def _key(value: float | None) -> str | None:
    return None if value is None else repr(value)


def _spark_exponent_mantissa(spark, values: list[float | None]) -> tuple[list, list]:
    p = f.col('p')
    exponent = f.log10(p).cast('int') - f.lit(1)
    rows = (
        spark
        .createDataFrame([(v,) for v in values], 'p DOUBLE')
        .select(exponent.alias('e'), f.round(p / f.pow(f.lit(10), exponent), 3).alias('m'))
        .collect()
    )
    return [r.e for r in rows], [_key(r.m) for r in rows]


def test_spark_log10_is_null_for_non_positive() -> None:
    out = pl.select(spark_log10(pl.lit(pl.Series([100.0, 0.0, -0.0, -1.0, math.nan, math.inf, None])))).to_series()
    assert [_key(v) for v in out.to_list()] == ['2.0', None, None, None, 'nan', 'inf', None]


def _ulps(value: float, steps: int) -> float:
    direction = math.inf if steps > 0 else 0.0
    for _ in range(abs(steps)):
        value = math.nextafter(value, direction)
    return value


def _edge(power: float, target: float, sign: int) -> float:
    """The farthest double from `power`, in `sign`'s direction, whose libm log10 is still `target`."""
    if math.log10(power) != target:
        return power
    step = 1
    while math.log10(_ulps(power, sign * step)) == target:
        step *= 2
    low, high = step // 2, step
    while high - low > 1:
        middle = (low + high) // 2
        if math.log10(_ulps(power, sign * middle)) == target:
            low = middle
        else:
            high = middle
    return _ulps(power, sign * low)


def _log10_edges() -> list[float]:
    values = []
    for k in range(324):
        power = float(f'1e{-k}')
        for edge in (_edge(power, -float(k), 1), _edge(power, -float(k), -1)):
            values += [_ulps(edge, step) for step in range(-4, 5)]
    rng = random.Random(20261018)
    return values + [10 ** rng.uniform(-323, 1) for _ in range(5000)]


def test_spark_log10_matches_spark_at_integer_edges(spark) -> None:
    values = _log10_edges()
    rows = spark.createDataFrame([(v,) for v in values], 'p DOUBLE').select(f.log10('p').alias('l')).collect()
    out = pl.DataFrame({'p': values}).select(spark_log10(pl.col('p')))
    mismatches = [
        (v, _key(r.l), _key(mine))
        for v, r, mine in zip(values, rows, out.to_series().to_list(), strict=True)
        if _key(r.l) != _key(mine)
    ]
    assert mismatches == []


def test_strict_pow_matches_spark(spark) -> None:
    exponents = [*range(-400, 401), 2**31 - 2, -(2**31), 2.0**40, -(2.0**40)]
    rows = (
        spark
        .createDataFrame([(float(k),) for k in exponents], 'k DOUBLE')
        .select(f.pow(f.lit(10), f.col('k')).alias('p'))
        .collect()
    )
    assert [_key(strict_pow(10.0, float(k))) for k in exponents] == [_key(r.p) for r in rows]


def test_exponent_mantissa_match_spark(spark) -> None:
    values = _pvalues()
    expected_exponent, expected_mantissa = _spark_exponent_mantissa(spark, values)
    out = pl.DataFrame({'p': values}, schema={'p': pl.Float64}).select(
        pvalue_exponent(pl.col('p')).alias('e'), pvalue_mantissa(pl.col('p')).alias('m')
    )
    assert out.schema == pl.Schema({'e': pl.Int32, 'm': pl.Float64})
    mismatches = [
        (p, e, m, se, sm)
        for p, e, m, se, sm in zip(
            values,
            out['e'].to_list(),
            [_key(v) for v in out['m'].to_list()],
            expected_exponent,
            expected_mantissa,
            strict=True,
        )
        if (e, m) != (se, sm)
    ]
    assert mismatches == []


def test_exponent_mantissa_review_focus() -> None:
    out = pl.DataFrame({'p': [0.0, 1e-8, 5e-320, math.nan, 1.0]}).select(
        pvalue_exponent(pl.col('p')).alias('e'), pvalue_mantissa(pl.col('p')).alias('m')
    )
    assert out['e'].to_list() == [None, -9, -320, -1, -1]
    assert [_key(v) for v in out['m'].to_list()] == [None, '10.0', '5.0', 'nan', '10.0']


ROUNDING = [
    1.0005,
    2.675,
    -1.0005,
    0.0005,
    0.00049,
    -0.0001,
    -0.0,
    9.9995,
    1.2345,
    1.0004999999999999,
    123456.7895,
    1e20,
    1.5e-5,
    5e-320,
    0.1 + 0.2,
    99.9999,
    math.nan,
    math.inf,
    -math.inf,
    None,
]


@pytest.mark.parametrize('scale', [3, 2, 0])
def test_round_half_up_matches_spark(spark, scale: int) -> None:
    rows = spark.createDataFrame([(v,) for v in ROUNDING], 'v DOUBLE').select(f.round('v', scale).alias('r')).collect()
    out = pl.select(round_half_up(pl.lit(pl.Series(ROUNDING, dtype=pl.Float64)), scale)).to_series()
    assert [_key(v) for v in out.to_list()] == [_key(r.r) for r in rows]


def test_round_half_up_uses_decimal_repr() -> None:
    out = pl.select(round_half_up(pl.lit(pl.Series([1.0005, 2.675, -1.0005, 0.00049, -0.0001])), 3)).to_series()
    assert [_key(v) for v in out.to_list()] == ['1.001', '2.675', '-1.001', '0.0', '0.0']
    assert pl.DataFrame({'v': [2.675]}).select(round_half_up(pl.col('v'), 2)).item() == 2.68


def test_normalise_mantissa() -> None:
    frame = pl.DataFrame(
        {'m': [10.0, 9.999, None, 10.001, math.nan], 'e': [-9, -9, None, -5, -1]},
        schema={'m': pl.Float64, 'e': pl.Int32},
    )
    mantissa, exponent = normalise_mantissa(pl.col('m'), pl.col('e'))
    out = frame.select(mantissa.alias('m'), exponent.alias('e'))
    assert out.schema == pl.Schema({'m': pl.Float64, 'e': pl.Int32})
    assert [_key(v) for v in out['m'].to_list()] == ['1.0', '9.999', None, repr(10.001 / 10), 'nan']
    assert out['e'].to_list() == [-8, -9, None, -4, -1]


@pytest.mark.parametrize(
    'pvalues',
    [
        [0.0, 1e-5, 3e-300, 0.0, 0.5, None],
        [1e-5, 3e-300, None],
        [0.0, 0.0],
        [0.0, -0.0, math.nan, 2e-10],
    ],
    ids=['zeros', 'no_zeros', 'all_zero', 'nan_and_negative_zero'],
)
@pytest.mark.parametrize('lazy', [False, True], ids=['eager', 'lazy'])
def test_substitute_zero_pvalues(spark, pvalues: list[float | None], lazy: bool) -> None:
    rows = _substitute_zero_pvalues(
        spark.createDataFrame([(i, v) for i, v in enumerate(pvalues)], 'i INT, p DOUBLE'), 'p', 'test'
    ).collect()
    frame = pl.DataFrame({'i': range(len(pvalues)), 'p': pvalues}, schema={'i': pl.Int32, 'p': pl.Float64})
    out = substitute_zero_pvalues(frame.lazy() if lazy else frame, 'p')
    assert isinstance(out, pl.LazyFrame) == lazy
    out = out.lazy().collect()
    assert out.schema == frame.schema
    assert [_key(v) for v in out['p'].to_list()] == [_key(r.p) for r in sorted(rows, key=lambda r: r.i)]


def test_substitute_zero_pvalues_logs_label() -> None:
    messages: list[str] = []
    sink = logger.add(messages.append, format='{message}')
    try:
        substitute_zero_pvalues(pl.DataFrame({'p': [0.0, 0.0, 1e-5]}), 'p', label='BRaVa Cauchy')
    finally:
        logger.remove(sink)
    assert messages == ['There are 2 BRaVa Cauchy evidence with a p-value of 0.0; they take the minimum, 1e-05.\n']
