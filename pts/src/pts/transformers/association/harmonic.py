"""Harmonic sum of evidence scores."""

from __future__ import annotations

from functools import reduce

import polars as pl

MAX_HARMONIC_SUM: float = reduce(lambda acc, iv: acc + iv[1] / iv[0] ** 2, enumerate([1] * 1000, start=1), 0.0)


def harmonic_sum(scores: pl.Expr, max_scores: int = 50) -> pl.Expr:
    """Sum of the top `max_scores` scores, sorted descending, each divided by its squared rank; 0.0 when empty."""
    ranked = pl.element().sort(descending=True).head(max_scores)
    rank = (pl.int_range(ranked.len()) + 1).cast(pl.Float64)
    return scores.list.eval((ranked / (rank * rank)).cum_sum().last()).list.first().fill_null(0.0)
