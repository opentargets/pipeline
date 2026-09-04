"""Rank constructions matching spark's window semantics.

Spark orders NULL as the smallest value: `ASC NULLS FIRST` and `DESC NULLS LAST` are both
consequences of that one rule. polars does neither -- `rank` returns NULL for a null input,
and a descending `sort` places nulls FIRST. Both differences silently change which rows
survive a `rank <= N` cut, so every rank in the search step goes through one of these two
helpers rather than calling `rank` directly.
"""

from __future__ import annotations

import polars as pl

#: Stands in for NULL when ranking. Spark treats NULL as smaller than every real value, which
#: is exactly what -inf does -- and it makes multiple nulls tie with each other, as spark's
#: rank does. Safe here because no score in this step is legitimately -inf.
_NULL_SORTS_SMALLEST = float('-inf')


def partitioned_rank(score: pl.Expr, *, by: str | list[str], descending: bool) -> pl.Expr:
    """Spark's `rank()` over `Window.partitionBy(by).orderBy(score)`.

    Nulls are mapped onto a sentinel that sorts where spark puts them, so the returned rank is
    never null and a `rank <= N` filter keeps the same rows spark kept.

    The `Float64` cast can also silently merge distinct large integers once they exceed 2**53
    (e.g. `2**60` and `2**60 + 1` round to the same float) -- harmless for this step's callers,
    whose scores are already a float (`transcriptScore`) or a small count (`credibleSetCount`).

    Args:
        score: the ordering expression. Cast to `Float64` internally.
        by: partition column(s).
        descending: True for `score.desc()`, False for `score.asc()`.

    Returns:
        An expression yielding a non-null `min`-method rank.
    """
    return (
        score.cast(pl.Float64)
        .fill_null(_NULL_SORTS_SMALLEST)
        .rank('min', descending=descending)
        .over(by)
    )


def composite_global_rank(lf: pl.LazyFrame, keys: list[str], *, alias: str = 'rank') -> pl.LazyFrame:
    """Spark's `rank()` over `Window.orderBy(k1.desc(), k2.desc())` -- global, multi-key.

    polars' `rank` takes a single expression and so cannot express a composite ordering at all.
    This builds it instead: sort by every key descending with spark's NULLS LAST placement,
    number the rows, then collapse ties by taking the smallest row number per distinct key
    tuple -- which is what the `min` rank method means.

    Args:
        lf: frame to rank.
        keys: ordering columns, all descending, most significant first.
        alias: name for the rank column.

    Returns:
        `lf` sorted by `keys` with a non-null `Int64` rank column added.
    """
    return (
        lf.sort(keys, descending=True, nulls_last=True)
        .with_row_index('_rank_row')
        .with_columns((pl.col('_rank_row').min().over(keys) + 1).cast(pl.Int64).alias(alias))
        .drop('_rank_row')
    )
