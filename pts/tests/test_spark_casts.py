"""Tests for the spark cast helpers."""

from __future__ import annotations

import math

import polars as pl
import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType, StructField, StructType

from pts.transformers.utils.spark_casts import (
    SPARK_CAST_TRIM,
    SPARK_INTEGER_CAST_TRIM,
    cast_double,
    to_float,
    to_int,
    to_long,
)

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


CAST_TEXT = [
    '1',
    ' 2 ',
    '3.7',
    '-4',
    'NA',
    '',
    '1e3',
    '2147483648',
    '0.5',
    '.',
    'x',
    '\t7\n',
    '-0.25',
    '\x7f8',
    '9\x7f',
    '+5',
    '-.5',
    '5.',
    '1.5d',
    '1_0',
    '0x10',
    None,
]


def test_integer_trim_set_adds_del() -> None:
    """Integer casts also trim DEL; float and double casts do not."""
    assert set(SPARK_INTEGER_CAST_TRIM) == set(SPARK_CAST_TRIM) | {'\x7f'}


def test_to_long_truncates_and_nulls() -> None:
    """Whole numbers are parsed, fractions truncated, anything else null."""
    df = pl.DataFrame({'s': ['3.7', '-4', ' 2 ', '.', '-.5', 'NA', '', None]}, schema={'s': pl.String})
    assert df.select(to_long(pl.col('s')))['s'].to_list() == [3, -4, 2, 0, 0, None, None, None]


def test_to_int_is_null_outside_int32() -> None:
    """Values beyond Int32 become null."""
    df = pl.DataFrame({'s': ['7', '2147483648']}, schema={'s': pl.String})
    assert df.select(to_int(pl.col('s')))['s'].to_list() == [7, None]


def test_to_float_is_float32() -> None:
    """Text parses to Float32 with null on bad text."""
    df = pl.DataFrame({'s': ['0.5', 'NA']}, schema={'s': pl.String})
    result = df.select(to_float(pl.col('s')))
    assert result.schema['s'] == pl.Float32
    assert result['s'].to_list() == [0.5, None]


def test_casts_match_spark(spark: SparkSession) -> None:
    """The integer and float casts equal spark's on tricky strings."""
    sdf = spark.createDataFrame([(t,) for t in CAST_TEXT], StructType([StructField('s', StringType())]))
    expected = [
        (r['l'], r['i'], r['f'])
        for r in sdf.select(
            sdf['s'].cast('long').alias('l'), sdf['s'].cast('int').alias('i'), sdf['s'].cast('float').alias('f')
        ).collect()
    ]
    result = pl.DataFrame({'s': CAST_TEXT}, schema={'s': pl.String}).select(
        to_long(pl.col('s')).alias('l'), to_int(pl.col('s')).alias('i'), to_float(pl.col('s')).alias('f')
    )
    assert [(r['l'], r['i'], r['f']) for r in result.to_dicts()] == expected
