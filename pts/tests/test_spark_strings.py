"""Tests for the polars equivalents of spark's string functions."""

import math

import polars as pl

from pts.transformers.utils.spark_strings import java_double_to_string, regexp_extract, spark_trim

DOUBLES = [
    0.0,
    -0.0,
    1.0,
    12.5,
    0.001,
    0.00099,
    9999999.0,
    1e7,
    1e-7,
    1.6688e-308,
    5.498e-320,
    1.7976931348623157e308,
    math.nan,
    math.inf,
    -math.inf,
    0.1 + 0.2,
    12.508500099182129,
    1e-5,
    -1.5e-10,
    -123456789.123,
    1e15,
    1e16,
    5.5410480786836218e17,
    2.3780253101263498e18,
    1e22,
]


def _spark_cast(spark, values: list[float]) -> list[str | None]:
    rows = spark.createDataFrame([(v,) for v in values], 'v DOUBLE').selectExpr('CAST(v AS STRING) AS s').collect()
    return [r['s'] for r in rows]


def _java(values: list[float | None]) -> list[str | None]:
    df = pl.DataFrame({'v': values}, schema={'v': pl.Float64})
    return df.select(java_double_to_string(pl.col('v')).alias('v'))['v'].to_list()


def test_spark_trim_strips_spaces_only() -> None:
    out = pl.select(spark_trim(pl.lit(pl.Series(['  a b  ', '\ta\t', '\xa0a\xa0', None])))).to_series()
    assert out.to_list() == ['a b', '\ta\t', '\xa0a\xa0', None]


def test_regexp_extract_is_empty_on_no_match_and_null_on_null() -> None:
    value = pl.lit(pl.Series(['PMID:123', 'none', None]))
    assert pl.select(regexp_extract(value, r'PMID:([0-9]+)', 1)).to_series().to_list() == ['123', '', None]


def test_java_double_to_string_matches_spark(spark) -> None:
    assert _java(DOUBLES) == _spark_cast(spark, DOUBLES)


def test_java_double_to_string_is_shortest_where_java_17_is_not(spark) -> None:
    assert _spark_cast(spark, [4.9e-324]) == ['4.9E-324']
    assert _java([4.9e-324]) == ['5.0E-324']


def test_java_double_to_string_keeps_null() -> None:
    assert _java([None, 1e-5]) == [None, '1.0E-5']
