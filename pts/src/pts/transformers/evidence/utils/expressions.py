"""Per-datasource score and direction-of-effect expressions for `evidence_postprocess`.

Hand-translated, native-polars replacement for the spark SQL strings `pts/config.yaml` carries in
`score_expression` / `direction_on_trait_expression` / `direction_on_target_expression` for each
`evidence_postprocess_*` step. Every expression below was diffed against real spark evaluating the
literal SQL from `config.yaml` -- see `task-registry-report.md` for the full parity run.

Keyed by `settings.datasource_id`, not the step name -- e.g. the `evidence_postprocess_clinvar`
step carries `datasource_id: eva`, so its entry lives under `'eva'`.

Several steps carry byte-for-byte or functionally identical expressions; those are written once as
a module-level function/constant and referenced from every entry that needs them, rather than
copy-pasted:

* `element_at(map(...), confidence)` is the shared shape behind `uniprot_variants`,
  `uniprot_literature` and others -- only the
  map differs, so `confidence_score` takes the map as its argument.
* `CASE WHEN diseaseId IS NOT NULL THEN '<value>' END` is the shared shape behind several direction
  expressions across `clinical_precedence`, `cosmic` and `intogen` -- `direction_when_disease_present` takes the
  literal as its argument.
* `CASE WHEN <col> == a THEN 'GoF' WHEN <col> == b THEN 'LoF' ELSE NULL END` is a 1:1 value lookup,
  behind `cancer_gene_census`'s `direction_on_target` -- expressed
  as `direction_lookup`, a `replace_strict` on the named column.
* The log10-rescaled-score shape (null/NaN stays null, a non-positive value saturates to 1.0,
  everything else linearly interpolated in log10 space and clamped) is shared by `intogen`,
  `gene_burden`, `crispr_screen` and the inner CASE of `expression_atlas` -- only the
  reference points and output column differ, so `log10_rescaled_score` takes them as arguments.

Provenance: `pts/config.yaml` no longer carries the original `score_expression` /
`direction_on_trait_expression` / `direction_on_target_expression` SQL strings -- git commit
`834f1db7` (just before they were stripped) has the last copy, one per `evidence_postprocess_*`
step, if you need to see the source SQL a given entry below was translated from. The translation
was verified 41/41 against real spark: on synthetic data covering the null/edge-case branches of
every expression, and separately on 20/20 real staged datasources end to end -- see
`task-registry-report.md` and `registry-parity-output.txt` for the runs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import polars as pl


def confidence_score(mapping: dict[str, float]) -> pl.Expr:
    """`element_at(map(...), confidence)` -- the shared shape across five datasources.

    `replace_strict` with `default=None` matches spark's `element_at`: a `confidence` value absent
    from `mapping` (or null) becomes null rather than raising.
    """
    return pl.col('confidence').replace_strict(mapping, default=None, return_dtype=pl.Float64)


def direction_when_disease_present(value: str) -> pl.Expr:
    """`CASE WHEN diseaseId IS NOT NULL THEN '<value>' END` -- shared by several direction fields."""
    return pl.when(pl.col('diseaseId').is_not_null()).then(pl.lit(value)).otherwise(None)


def direction_lookup(column: str, mapping: dict[str, str]) -> pl.Expr:
    """`CASE WHEN <column> == a THEN x WHEN <column> == b THEN y ELSE NULL END` as a 1:1 lookup."""
    return pl.col(column).replace_strict(mapping, default=None)


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


@dataclass(frozen=True)
class DatasourceExpressions:
    """The score and direction-of-effect expressions for one `datasourceId`."""

    score: pl.Expr
    direction_on_trait: pl.Expr | None = None
    direction_on_target: pl.Expr | None = None


EXPRESSIONS: dict[str, DatasourceExpressions] = {}
