"""Score expression helpers shared by the evidence datasource modules."""

from __future__ import annotations

import polars as pl

from pts.transformers.utils.spark_math import divide, spark_log10


def confidence_score(mapping: dict[str, float]) -> pl.Expr:
    """`element_at(map(...), confidence)` -- the shared shape across datasources.

    `replace_strict` with `default=None` matches spark's `element_at`: a `confidence` value absent
    from `mapping` (or null) becomes null rather than raising.
    """
    return pl.col('confidence').replace_strict(mapping, default=None, return_dtype=pl.Float64)


def direction_when_disease_present(value: str) -> pl.Expr:
    """`CASE WHEN diseaseId IS NOT NULL THEN '<value>' END` -- shared by several direction fields."""
    return pl.when(pl.col('diseaseId').is_not_null()).then(pl.lit(value)).otherwise(None)


def log10_rescaled_score(value: pl.Expr, out_min: float, weak_ref: float, strong_ref: float) -> pl.Expr:
    """Rescale a p-value-like `value` onto `[out_min, 1.0]` in log10 space, clamped.

    Common shape behind the intogen/crispr_screen score expressions, and
    expression_atlas's inner CASE:

    * null or NaN input stays null.
    * a non-positive value saturates to the strongest score, 1.0.
    * otherwise, linearly interpolate in log10 space (spark's fdlibm `log10`) between `weak_ref`
      (-> `out_min`) and `strong_ref` (-> 1.0), then clamp back to `[out_min, 1.0]` for values
      past either reference point -- spark's `LEAST(1.0, GREATEST(out_min, ...))`.

    Args:
        value: the column to rescale.
        out_min: the output score for `value == weak_ref` (and the clamp floor).
        weak_ref: the value that scores `out_min`.
        strong_ref: the value that scores `1.0`.
    """
    refs = pl.Series([weak_ref, strong_ref], dtype=pl.Float64)
    log_weak, log_strong = pl.select(spark_log10(pl.lit(refs))).to_series().to_list()
    rescaled = divide((1.0 - out_min) * (spark_log10(value) - log_weak), log_strong - log_weak) + out_min
    return (
        pl
        .when(value.is_null() | value.is_nan())
        .then(None)
        .when(value <= 0)
        .then(1.0)
        .otherwise(rescaled.clip(out_min, 1.0))
    )
