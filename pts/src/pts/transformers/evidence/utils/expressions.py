"""Score expression helpers shared by the evidence datasource modules."""

from __future__ import annotations

import math

import numpy as np
import polars as pl


def confidence_score(mapping: dict[str, float]) -> pl.Expr:
    """`element_at(map(...), confidence)` -- the shared shape across datasources.

    `replace_strict` with `default=None` matches spark's `element_at`: a `confidence` value absent
    from `mapping` (or null) becomes null rather than raising.
    """
    return pl.col('confidence').replace_strict(mapping, default=None, return_dtype=pl.Float64)


def direction_when_disease_present(value: str) -> pl.Expr:
    """`CASE WHEN diseaseId IS NOT NULL THEN '<value>' END` -- shared by several direction fields."""
    return pl.when(pl.col('diseaseId').is_not_null()).then(pl.lit(value)).otherwise(None)


def _true_divide(series: pl.Series, divisor: float) -> pl.Series:
    values = series.cast(pl.Float64).to_numpy() / divisor
    out = pl.Series(series.name, values, dtype=pl.Float64, nan_to_null=False)
    nulls = np.flatnonzero(series.is_null().to_numpy())
    return out.scatter(nulls, None) if len(nulls) else out


def divide(expr: pl.Expr, divisor: float) -> pl.Expr:
    """Divide by a constant with IEEE division, keeping nulls and NaN.

    Polars 2 does not divide exactly: `x / 10` differs from spark in the last bit for about a third
    of values.
    """
    return expr.map_batches(lambda s: _true_divide(s, divisor), return_dtype=pl.Float64, is_elementwise=True)


def _true_log10(series: pl.Series) -> pl.Series:
    with np.errstate(divide='ignore', invalid='ignore'):
        values = np.log10(series.cast(pl.Float64).to_numpy())
    out = pl.Series(series.name, values, dtype=pl.Float64, nan_to_null=False)
    nulls = np.flatnonzero(series.is_null().to_numpy())
    return out.scatter(nulls, None) if len(nulls) else out


def log10(expr: pl.Expr) -> pl.Expr:
    """Base-10 logarithm through numpy, keeping nulls and NaN.

    Polars 2 `Expr.log10` differs from libm in the last bit for about half of all values.
    """
    return expr.map_batches(_true_log10, return_dtype=pl.Float64, is_elementwise=True)


def log10_rescaled_score(value: pl.Expr, out_min: float, weak_ref: float, strong_ref: float) -> pl.Expr:
    """Rescale a p-value-like `value` onto `[out_min, 1.0]` in log10 space, clamped.

    Common shape behind the intogen/gene_burden/crispr_screen score expressions, and
    expression_atlas's inner CASE:

    * null or NaN input stays null.
    * a non-positive value saturates to the strongest score, 1.0.
    * otherwise, linearly interpolate in log10 space between `weak_ref` (-> `out_min`) and
      `strong_ref` (-> 1.0), then clamp back to `[out_min, 1.0]` for values past either reference
      point -- spark's `LEAST(1.0, GREATEST(out_min, ...))`.

    Args:
        value: the column to rescale.
        out_min: the output score for `value == weak_ref` (and the clamp floor).
        weak_ref: the value that scores `out_min`.
        strong_ref: the value that scores `1.0`.
    """
    log_weak = math.log10(weak_ref)
    log_strong = math.log10(strong_ref)
    rescaled = divide((1.0 - out_min) * (log10(value) - log_weak), log_strong - log_weak) + out_min
    return (
        pl
        .when(value.is_null() | value.is_nan())
        .then(None)
        .when(value <= 0)
        .then(1.0)
        .otherwise(rescaled.clip(out_min, 1.0))
    )
