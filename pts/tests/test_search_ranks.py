"""Tests for the search rank helpers.

Spark treats NULL as the smallest value when ordering: `ASC NULLS FIRST`, `DESC NULLS LAST`.
polars instead returns a NULL rank for a null input and sorts nulls FIRST on a descending
sort. Both differences change which rows survive a `rank <= N` cut, so both are pinned here.
"""

import polars as pl

from pts.transformers.search.ranks import composite_global_rank, partitioned_rank


def test_partitioned_rank_gives_tied_scores_the_same_rank_and_skips_the_next() -> None:
    frame = pl.DataFrame({'g': ['x'] * 3, 's': [5.0, 1.0, 1.0]})

    result = frame.with_columns(partitioned_rank(pl.col('s'), by='g', descending=True).alias('r'))

    assert result['r'].to_list() == [1, 2, 2]


def test_partitioned_rank_skips_the_rank_after_a_tie() -> None:
    """A tie followed by a non-tied value is the only shape that distinguishes `min` rank from
    `dense` rank. `min` skips the ranks consumed by the tie (1, 2, 2, 4); `dense` would give the
    last row `3` instead."""
    frame = pl.DataFrame({'g': ['x'] * 4, 's': [5.0, 1.0, 1.0, 0.0]})

    result = frame.with_columns(partitioned_rank(pl.col('s'), by='g', descending=True).alias('r'))

    assert result['r'].to_list() == [1, 2, 2, 4]


def test_partitioned_rank_is_independent_per_partition() -> None:
    frame = pl.DataFrame({'g': ['x', 'x', 'y'], 's': [5.0, 1.0, 3.0]})

    result = frame.with_columns(partitioned_rank(pl.col('s'), by='g', descending=True).alias('r'))

    assert result['r'].to_list() == [1, 2, 1]


def test_spark_ascending_rank_puts_nulls_first() -> None:
    """Spark's `orderBy(col.asc())` is NULLS FIRST, so a null-scored row ranks 1 and SURVIVES
    a `rank <= 50` cut. polars' bare `rank` returns null there, and the row is silently
    dropped by the same filter."""
    frame = pl.DataFrame({'g': ['x'] * 3, 's': [None, 5.0, 1.0]})

    result = frame.with_columns(partitioned_rank(pl.col('s'), by='g', descending=False).alias('r'))

    assert result['r'].to_list() == [1, 3, 2]
    assert result.filter(pl.col('r') <= 2).height == 2


def test_spark_descending_rank_puts_nulls_last() -> None:
    frame = pl.DataFrame({'g': ['x'] * 3, 's': [None, 5.0, 1.0]})

    result = frame.with_columns(partitioned_rank(pl.col('s'), by='g', descending=True).alias('r'))

    assert result['r'].to_list() == [3, 1, 2]


def test_partitioned_rank_never_returns_null() -> None:
    frame = pl.DataFrame({'g': ['x'] * 2, 's': [None, None]})

    result = frame.with_columns(partitioned_rank(pl.col('s'), by='g', descending=True).alias('r'))

    assert result['r'].null_count() == 0
    assert result['r'].to_list() == [1, 1]


def test_composite_global_rank_orders_on_two_keys_descending() -> None:
    frame = pl.LazyFrame({'id': ['a', 'b', 'c'], 'c': [1.0, 3.0, 3.0], 'n': [99, 7, 9]})

    result = composite_global_rank(frame, ['c', 'n']).collect().sort('id')

    assert dict(zip(result['id'], result['rank'], strict=True)) == {'a': 3, 'b': 2, 'c': 1}


def test_composite_global_rank_ties_share_a_rank_and_skip_the_next() -> None:
    frame = pl.LazyFrame({'id': ['a', 'b', 'c'], 'c': [3.0, 3.0, 1.0], 'n': [7, 7, 5]})

    result = composite_global_rank(frame, ['c', 'n']).collect().sort('id')

    assert dict(zip(result['id'], result['rank'], strict=True)) == {'a': 1, 'b': 1, 'c': 3}


def test_spark_composite_global_rank_puts_nulls_last() -> None:
    """`credibleSetCount` is null for every study with no credible set. Spark's DESC NULLS
    LAST ranks those worst; polars sorts nulls FIRST by default, which would invert the
    multiplier for a large share of the 4.07M studies."""
    frame = pl.LazyFrame({'id': ['a', 'b'], 'c': [None, 1.0], 'n': [10, 10]})

    result = composite_global_rank(frame, ['c', 'n']).collect().sort('id')

    assert dict(zip(result['id'], result['rank'], strict=True)) == {'a': 2, 'b': 1}


def test_composite_global_rank_never_returns_null() -> None:
    frame = pl.LazyFrame({'id': ['a', 'b'], 'c': [None, None], 'n': [None, 3]})

    result = composite_global_rank(frame, ['c', 'n']).collect()

    assert result['rank'].null_count() == 0
