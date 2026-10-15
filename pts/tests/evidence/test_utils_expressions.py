"""Tests for the shared score expression helpers."""

from __future__ import annotations

import math
import random
import warnings
from math import isnan

import numpy as np
import polars as pl
from pyspark.sql import SparkSession
from pyspark.sql import functions as f

from pts.transformers.evidence.encore import ENCORE
from pts.transformers.evidence.utils.expressions import log10


def test_log10_is_numpy_log10_and_within_one_ulp_of_libm() -> None:
    rng = random.Random(0)
    values = [10 ** rng.uniform(-12, 3) for _ in range(5000)]
    got = pl.DataFrame({'x': values}).select(log10(pl.col('x')).alias('y'))['y'].to_list()
    assert got == np.log10(np.array(values)).tolist()
    assert all(abs(g - math.log10(v)) <= math.ulp(math.log10(v)) for g, v in zip(got, values, strict=True))


def test_log10_keeps_null_nan_zero_and_one() -> None:
    edge = pl.DataFrame({'x': [None, float('nan'), 0.0, 1.0]}).select(log10(pl.col('x')).alias('y'))['y'].to_list()
    assert edge[0] is None
    assert isnan(edge[1])
    assert edge[2] == float('-inf')
    assert edge[3] == 0.0


def test_log10_accepts_integer_and_float32_input() -> None:
    ints = pl.DataFrame({'x': [1, 10, 1000, None]}, schema={'x': pl.Int64}).select(log10(pl.col('x')).alias('y'))['y']
    assert ints.dtype == pl.Float64
    assert ints.to_list() == [0.0, 1.0, 3.0, None]
    floats = pl.DataFrame({'x': [10.0, 100.0]}, schema={'x': pl.Float32}).select(log10(pl.col('x')).alias('y'))['y']
    assert floats.dtype == pl.Float64
    assert floats.to_list() == [1.0, 2.0]


def test_log10_of_a_negative_is_nan_without_a_warning() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        got = pl.DataFrame({'x': [-1.0, 0.0]}).select(log10(pl.col('x')).alias('y'))['y'].to_list()
    assert isnan(got[0])
    assert got[1] == float('-inf')


def test_log10_of_an_all_null_column_is_all_null() -> None:
    got = pl.DataFrame({'x': [None, None]}, schema={'x': pl.Float64}).select(log10(pl.col('x')).alias('y'))['y']
    assert got.to_list() == [None, None]


# ------------------------------------------------------------------------------------------------
# Exact parity with the SQL on main for the scores rewritten since the case table was measured.
# ------------------------------------------------------------------------------------------------

ENCORE_SQL = 'ABS ( geneticInteractionScore  ) / 16.0'


def _spark_scores(spark: SparkSession, column: str, values: list, ddl_type: str, sql: str) -> list:
    df = spark.createDataFrame([(v,) for v in values], f'{column} {ddl_type}')
    return [row.s for row in df.select(f.expr(sql).cast('double').alias('s')).collect()]


def _same(a: list, b: list) -> bool:
    return len(a) == len(b) and all(
        (x is None and y is None) or (x is not None and y is not None and (x == y or (isnan(x) and isnan(y))))
        for x, y in zip(a, b, strict=True)
    )


def test_encore_score_matches_spark_bit_for_bit(spark: SparkSession) -> None:
    rng = random.Random(0)
    values = [0.0, -0.0, 16.0, -32.5, 1e-310, float('inf'), float('nan'), None]
    values += [rng.uniform(-20.0, 20.0) for _ in range(5000)]
    expected = _spark_scores(spark, 'geneticInteractionScore', values, 'DOUBLE', ENCORE_SQL)
    got = pl.DataFrame({'geneticInteractionScore': values}, schema={'geneticInteractionScore': pl.Float64})
    assert _same(got.select(ENCORE.score.alias('s'))['s'].to_list(), expected)
