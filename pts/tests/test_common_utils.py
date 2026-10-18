"""Tests for pts.pyspark.common.utils helpers."""


from pts.pyspark.common.utils import maybe_coalesce, maybe_repartition

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
