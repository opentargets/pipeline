"""Tests for the shared score expression helpers."""

from __future__ import annotations

import random
from math import isnan

import polars as pl
import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as f

from pts.transformers.evidence.encore import ENCORE
from pts.transformers.evidence.utils.expressions import log10_rescaled_score

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


SCORE_PARAMETERS = [(0.25, 1e-7, 1e-17), (0.0, 0.5, 0.005), (0.25, 0.1, 1e-10), (0.0, 1.0, 1e-10)]


@pytest.mark.parametrize(('out_min', 'weak_ref', 'strong_ref'), SCORE_PARAMETERS)
def test_log10_rescaled_score_matches_spark_bit_for_bit(
    spark: SparkSession, out_min: float, weak_ref: float, strong_ref: float
) -> None:
    rng = random.Random(0)
    values = [None, float('nan'), float('inf'), 0.0, -0.0, -1.0, 1.0, 0.5, 1e-310, weak_ref, strong_ref, 1e-300]
    values += [10 ** rng.uniform(-300, 0) for _ in range(3000)]
    values += [10 ** rng.uniform(-20, 0) for _ in range(3000)]
    sql = (
        f'CASE WHEN p IS NULL OR isnan(p) THEN NULL WHEN p <= 0 THEN 1.0 ELSE '
        f'LEAST(1.0, GREATEST({out_min!r}, (1.0 - {out_min!r}) * (log10(p) - log10(CAST({weak_ref!r} AS DOUBLE))) '
        f'/ (log10(CAST({strong_ref!r} AS DOUBLE)) - log10(CAST({weak_ref!r} AS DOUBLE))) + {out_min!r})) END'
    )
    expected = _spark_scores(spark, 'p', values, 'DOUBLE', sql)
    got = pl.DataFrame({'p': values}, schema={'p': pl.Float64})
    score = log10_rescaled_score(pl.col('p'), out_min, weak_ref, strong_ref)
    assert _same(got.select(score.alias('s'))['s'].to_list(), expected)
