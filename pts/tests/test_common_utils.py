"""Tests for pts.pyspark.common.utils helpers."""

import numpy as np
import pandas as pd

from pts.pyspark.common.utils import maybe_coalesce, maybe_repartition, pandas_to_spark_strings

# ---------------------------------------------------------------------------
# maybe_coalesce
# ---------------------------------------------------------------------------


def test_maybe_coalesce_when_count_given(spark):
    df = spark.range(100).repartition(8)
    result = maybe_coalesce(df, 2)
    assert result.rdd.getNumPartitions() == 2  # ty:ignore[call-non-callable]


def test_maybe_coalesce_returns_df_unchanged_when_none(spark):
    df = spark.range(100)
    result = maybe_coalesce(df, None)
    assert result is df


def test_maybe_coalesce_returns_df_unchanged_when_zero(spark):
    df = spark.range(100)
    result = maybe_coalesce(df, 0)
    assert result is df


# ---------------------------------------------------------------------------
# maybe_repartition
# ---------------------------------------------------------------------------


def test_maybe_repartition_when_count_given(spark):
    df = spark.range(100)
    result = maybe_repartition(df, 4)
    assert result.rdd.getNumPartitions() == 4  # ty:ignore[call-non-callable]


def test_maybe_repartition_returns_df_unchanged_when_none(spark):
    df = spark.range(100)
    result = maybe_repartition(df, None)
    assert result is df


def test_maybe_repartition_returns_df_unchanged_when_zero(spark):
    df = spark.range(100)
    result = maybe_repartition(df, 0)
    assert result is df


# ---------------------------------------------------------------------------
# pandas_to_spark_strings
# ---------------------------------------------------------------------------


def test_pandas_to_spark_strings_types_every_cell_as_a_string(spark):
    pdf = pd.DataFrame({
        'mixed': [1.5, '-', None],
        'integer': [1, 2, 3],
        'text': ['a', None, 'c'],
        'flag': pd.array([True, False, None], dtype='boolean'),
        'numpy_flag': np.array([True, False, True]),
        'object_flag': [True, None, False],
        'unused': [1.0, 'x', None],
    })

    df = pandas_to_spark_strings(spark, pdf, ['mixed', 'integer', 'text', 'flag', 'numpy_flag', 'object_flag'])

    assert all(dtype == 'string' for _, dtype in df.dtypes)
    assert [r.asDict() for r in df.collect()] == [
        {'mixed': '1.5', 'integer': '1', 'text': 'a', 'flag': '1', 'numpy_flag': '1', 'object_flag': '1'},
        {'mixed': '-', 'integer': '2', 'text': None, 'flag': '0', 'numpy_flag': '0', 'object_flag': None},
        {'mixed': None, 'integer': '3', 'text': 'c', 'flag': None, 'numpy_flag': '1', 'object_flag': '0'},
    ]
