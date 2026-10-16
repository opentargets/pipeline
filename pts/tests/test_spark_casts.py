"""Tests for the spark cast helpers."""

from __future__ import annotations

import math

import polars as pl
import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType, StructField, StructType

from pts.transformers.utils.spark_casts import SPARK_CAST_TRIM, cast_double

TRICKY = [
    ' 1.5 ',
    '\x7f2',
    '\t3',
    '1e3',
    '1.5d',
    '2F',
    '.5',
    '5.',
    '1_0',
    '0x10',
    'NaN',
    '+NaN',
    '-NaN',
    '-nan',
    '+nan',
    'nan',
    'Infinity',
    '-Infinity',
    'inf',
    '',
    'abc',
    None,
]


def _same(a: float | None, b: float | None) -> bool:
    if a is None or b is None:
        return a is b
    return (math.isnan(a) and math.isnan(b)) or a == b


def test_trim_set() -> None:
    """The trim set is every character up to and including space, and not DEL."""
    assert set(SPARK_CAST_TRIM) == {chr(c) for c in range(0x21)}


@pytest.mark.parametrize(
    ('text', 'expected'),
    [(' 1.5 ', 1.5), ('1e3', 1000.0), ('1.5d', 1.5), ('abc', None), ('', None), (None, None), ('-nan', None)],
)
def test_cast_double(text: str | None, expected: float | None) -> None:
    """Text is cast as spark does, with null on bad text."""
    df = pl.DataFrame({'x': [text]}, schema={'x': pl.String})
    assert df.select(cast_double('x'))['x'].to_list() == [expected]


def test_cast_double_matches_spark(spark: SparkSession) -> None:
    """The polars cast equals spark's on tricky strings."""
    sdf = spark.createDataFrame([(t,) for t in TRICKY], StructType([StructField('x', StringType())]))
    expected = [r['x'] for r in sdf.select(sdf['x'].cast('double').alias('x')).collect()]
    result = pl.DataFrame({'x': TRICKY}, schema={'x': pl.String}).select(cast_double('x'))['x'].to_list()
    mismatches = [(t, r, e) for t, r, e in zip(TRICKY, result, expected, strict=True) if not _same(r, e)]
    assert not mismatches
