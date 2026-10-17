"""Tests for Spark's frame operations in polars, pinned against Spark."""

from __future__ import annotations

import math

import polars as pl
from pyspark.sql import SparkSession

from pts.transformers.utils.spark_frames import spark_distinct

ROWS = [
    ('a', -0.0, 1.0),
    ('a', 0.0, 1.0),
    ('b', 0.0, -0.0),
    ('b', math.nan, None),
    ('b', math.nan, None),
    ('c', -1.5, 2.0),
    ('c', -1.5, 2.0),
    ('a', -0.0, 1.0),
]


def test_spark_distinct_matches_spark(spark: SparkSession) -> None:
    expected = spark.createDataFrame(ROWS, 'k string, x double, y double').distinct().collect()
    frame = pl.DataFrame(ROWS, schema={'k': pl.String, 'x': pl.Float64, 'y': pl.Float64}, orient='row')
    assert sorted(map(repr, spark_distinct(frame).rows())) == sorted(repr(tuple(r)) for r in expected)


def test_spark_distinct_keeps_the_first_in_order_eager_and_lazy() -> None:
    frame = pl.DataFrame(ROWS, schema={'k': pl.String, 'x': pl.Float32, 'y': pl.Float64}, orient='row')
    eager = spark_distinct(frame)
    assert repr(eager.rows()) == repr(spark_distinct(frame.lazy()).collect().rows())
    assert eager['k'].to_list() == ['a', 'b', 'b', 'c']
    assert eager.schema == frame.schema
