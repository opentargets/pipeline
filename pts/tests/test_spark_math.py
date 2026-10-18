"""Tests for the polars equivalents of spark's p-value maths."""

import math
import random

import polars as pl
import pyspark.sql.functions as f
import pytest
from loguru import logger

from pts.transformers.utils.spark_math import (
    normalise_mantissa,
    pvalue_exponent,
    pvalue_mantissa,
    round_half_up,
    spark_double_to_int,
    spark_exp,
    spark_log10,
    spark_pow,
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


def _log10(value: float) -> float:
    return pl.select(spark_log10(pl.lit(value, dtype=pl.Float64))).item()


def _edge(power: float, target: float, sign: int) -> float:
    """The farthest double from `power`, in `sign`'s direction, whose `spark_log10` is still `target`."""
    if _log10(power) != target:
        return power
    step = 1
    while _log10(_ulps(power, sign * step)) == target:
        step *= 2
    low, high = step // 2, step
    while high - low > 1:
        middle = (low + high) // 2
        if _log10(_ulps(power, sign * middle)) == target:
            low = middle
        else:
            high = middle
    return _ulps(power, sign * low)


def _log10_edges() -> list[float]:
    values = []
    for k in range(324):
        power = float(f'1e{-k}')
        for sign in (1, -1):
            edge = _edge(power, -float(k), sign)
            values += [edge, _ulps(edge, sign)]
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


def _pow_exponents() -> list[float | None]:
    rng = random.Random(20261019)
    values: list[float | None] = [float(f'{rng.uniform(0, 400):.{rng.randint(1, 9)}f}') for _ in range(20000)]
    values += [float(f'{rng.uniform(1, 10):.5f}e-{rng.randint(1, 9)}') for _ in range(2000)]
    values += [rng.uniform(-1100, 1100) for _ in range(2000)]
    values += [0.0, -0.0, 1.0, -1.0, 2.0, 0.5, -0.5, 400.0, 323.3, 323.31, 307.5, 307.65, 90.0033, 9.36597e-08]
    values += [2.0**31, -(2.0**31), 2.0**31 + 1, 1e300, -1e300, 5e-324, math.nan, math.inf, -math.inf, None]
    return [None if v is None else -v for v in values]


@pytest.mark.parametrize('base', [10.0, 2.0, 0.3])
def test_spark_pow_matches_spark(spark, base: float) -> None:
    exponents = _pow_exponents()
    rows = (
        spark
        .createDataFrame([(v,) for v in exponents], 'y DOUBLE')
        .select(f.pow(f.lit(base), f.col('y')).alias('p'))
        .collect()
    )
    out = pl.DataFrame({'y': exponents}, schema={'y': pl.Float64}).select(spark_pow(base, pl.col('y')))
    assert out.schema == pl.Schema({'y': pl.Float64})
    mismatches = [
        (y, _key(r.p), _key(mine))
        for y, r, mine in zip(exponents, rows, out.to_series().to_list(), strict=True)
        if _key(r.p) != _key(mine)
    ]
    assert mismatches == []


def test_spark_pow_is_not_numpy_power() -> None:
    # 10 ** -23.6442024: numpy's power is one ulp above fdlibm's
    assert pl.select(spark_pow(10.0, pl.lit(-23.6442024))).item() == 2.268807243217692e-24
    assert strict_pow(10.0, -23.6442024) == 2.268807243217692e-24


@pytest.mark.parametrize('base', [1 + 2**-30, 1 - 2**-22])
def test_strict_pow_raises_for_a_huge_exponent_on_a_base_near_one(base: float) -> None:
    with pytest.raises(ValueError, match='unsupported exponent'):
        strict_pow(base, 2.0**40)
    assert strict_pow(base, 2.0**70) == (math.inf if base > 1 else 0.0)
    assert strict_pow(base, -(2.0**70)) == (0.0 if base > 1 else math.inf)


def _exp_arguments() -> list[float | None]:
    rng = random.Random(20261020)
    values: list[float | None] = [
        float(f'{rng.uniform(-20, 20):.{rng.randint(1, 9)}f}') for _ in range(20000)
    ]
    values += [rng.uniform(-750, 712) for _ in range(5000)]
    values += [rng.uniform(-1.1, 1.1) for _ in range(2000)]
    values += [0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 0.3465735902799726, 0.34657359027997264, 1.0397207708399179]
    values += [1e-9, -1e-9, 3.7e-9, 1e-300, 5e-324, 7.09782712893383973096e02, 709.79, -745.13321910194110842]
    values += [-745.2, -708.4, -708.3964185322641, -720.5, 2.0**-28, -(2.0**-28), math.nan, math.inf, -math.inf, None]
    return values


def test_spark_exp_matches_spark(spark) -> None:
    arguments = _exp_arguments()
    rows = spark.createDataFrame([(v,) for v in arguments], 'x DOUBLE').select(f.exp('x').alias('e')).collect()
    out = pl.DataFrame({'x': arguments}, schema={'x': pl.Float64}).select(spark_exp(pl.col('x')))
    assert out.schema == pl.Schema({'x': pl.Float64})
    mismatches = [
        (x, _key(r.e), _key(mine))
        for x, r, mine in zip(arguments, rows, out.to_series().to_list(), strict=True)
        if _key(r.e) != _key(mine)
    ]
    assert mismatches == []


def test_spark_exp_is_not_numpy_exp() -> None:
    # exp(2.1): numpy's is one ulp above fdlibm's
    assert pl.select(spark_exp(pl.lit(2.1))).item() == 8.16616991256765


def test_spark_double_to_int_matches_spark(spark) -> None:
    values: list[float | None] = [0.0, -0.0, 0.9, -0.9, 10.0, 2147483647.9, 2147483648.0, -2147483649.0]
    values += [3e9, -3e9, 1e300, math.nan, math.inf, -math.inf, None]
    rows = spark.createDataFrame([(v,) for v in values], 'x DOUBLE').select(f.col('x').cast('int').alias('i')).collect()
    out = pl.DataFrame({'x': values}, schema={'x': pl.Float64}).select(spark_double_to_int(pl.col('x')))
    assert out.schema == pl.Schema({'x': pl.Int32})
    assert out.to_series().to_list() == [r.i for r in rows]


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


def test_round_half_up_broadcasts_a_scalar() -> None:
    assert pl.select(round_half_up(pl.lit(1.0005), 3)).to_series().to_list() == [1.001]


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
    ('pvalues', 'expected'),
    [
        ([0.0, 1e-5, 3e-300, 0.0, 0.5, None], ['3e-300', '1e-05', '3e-300', '3e-300', '0.5', None]),
        ([1e-5, 3e-300, None], ['1e-05', '3e-300', None]),
        ([0.0, 0.0], [None, None]),
        ([0.0, -0.0, math.nan, 2e-10], ['2e-10', '2e-10', 'nan', '2e-10']),
    ],
    ids=['zeros', 'no_zeros', 'all_zero', 'nan_and_negative_zero'],
)
@pytest.mark.parametrize('lazy', [False, True], ids=['eager', 'lazy'])
def test_substitute_zero_pvalues(pvalues: list[float | None], expected: list[str | None], lazy: bool) -> None:
    frame = pl.DataFrame({'i': range(len(pvalues)), 'p': pvalues}, schema={'i': pl.Int32, 'p': pl.Float64})
    out = substitute_zero_pvalues(frame.lazy() if lazy else frame, 'p')
    assert isinstance(out, pl.LazyFrame) == lazy
    out = out.lazy().collect()
    assert out.schema == frame.schema
    # expected values captured from the removed pyspark implementation
    assert [_key(v) for v in out['p'].to_list()] == expected


def test_substitute_zero_pvalues_logs_label() -> None:
    messages: list[str] = []
    sink = logger.add(messages.append, format='{message}')
    try:
        substitute_zero_pvalues(pl.DataFrame({'p': [0.0, 0.0, 1e-5]}), 'p', label='BRaVa Cauchy')
    finally:
        logger.remove(sink)
    assert messages == ['There are 2 BRaVa Cauchy evidence with a p-value of 0.0; they take the minimum, 1e-05.\n']


def test_substitute_zero_pvalues_counts_and_takes_the_minimum_over_the_scope() -> None:
    scope = pl.LazyFrame({'p': [0.0, 0.5, 0.0, 1e-30, 1e-5]})
    frame = scope.filter((pl.col('p') <= 1e-4) | (pl.col('p') == 0.0)).filter(pl.col('p') != 1e-30)
    messages: list[str] = []
    sink = logger.add(messages.append, format='{message}')
    try:
        out = substitute_zero_pvalues(frame, 'p', label='Genebass', scope=scope)
    finally:
        logger.remove(sink)
    assert isinstance(out, pl.LazyFrame)
    assert out.collect()['p'].to_list() == [1e-30, 1e-30, 1e-5]
    assert messages == ['There are 2 Genebass evidence with a p-value of 0.0; they take the minimum, 1e-30.\n']
