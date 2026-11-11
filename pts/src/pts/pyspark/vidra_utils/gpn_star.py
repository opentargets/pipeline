"""GPN-Star for VIDRA: the benign-probability transform and its ClinVar calibration.

VIDRA's protein-function annotations are on a `[0, 1]` scale where 1 is benign and 0 is damaging.
The variant index carries the raw GPN-Star calibrated log-likelihood ratio (`variantEffect.score`
of method `GPN-Star`; the more negative, the more constrained) and a quantile-binned
`normalisedScore` that is 0 for the least constrained 90% of the genome. The binned score throws
away the information VIDRA needs on missense variants, so VIDRA maps the raw LLR through a
two-parameter logistic instead:

    as_gpnstar = 1 - sigmoid(a + b * llr)

`a` and `b` are fitted once per variant-index release by `calibrate`: a logistic regression of
ClinVar pathogenic/likely-pathogenic against benign/likely-benign missense variants on their LLR,
at ClinVar's own class balance, so that `sigmoid(a + b * llr)` is the probability that a ClinVar
missense variant with that score is pathogenic. The fitted values are stored in the `vidra_input`
step settings next to the release they were fitted on.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import polars as pl
from scipy.optimize import minimize
from scipy.stats import rankdata

MISSENSE = 'SO_0001583'
PATHOGENIC_LABELS = ('pathogenic', 'likely pathogenic')
BENIGN_LABELS = ('benign', 'likely benign')
REVIEWED_CONFIDENCE = ('reviewed by expert panel', 'practice guideline')
"""Review statuses above the star-rated ones, always kept."""


def benign_score(llr: pl.Expr, a: float, b: float) -> pl.Expr:
    """`1 - sigmoid(a + b * llr)`: the VIDRA GPN-Star annotation, 1 benign and 0 damaging; null stays null."""
    return 1.0 - 1.0 / (1.0 + (-(a + b * llr.cast(pl.Float64))).exp())


def clinvar_missense_labels(evidence_eva: pl.LazyFrame, *, min_review: str = 'single') -> pl.LazyFrame:
    """One label per ClinVar missense variant: 1 pathogenic/likely pathogenic, 0 benign/likely benign.

    Rows are EVA evidence with `variantFunctionalConsequenceId` missense. Submissions without
    assertion criteria, without a classification, or with conflicting classifications are dropped
    (`confidence`); `min_review='multiple'` further keeps only variants with multiple submitters and
    no conflicts, expert panels and practice guidelines. A variant with both a pathogenic and a
    benign label across its submissions is dropped. `targetId` is the gene of the first kept row,
    for the AlphaMissense comparison.

    Examples:
        >>> eva = pl.LazyFrame({
        ...     'variantId': ['v1', 'v1', 'v2', 'v3', 'v3', 'v4'],
        ...     'targetId': ['g'] * 6,
        ...     'variantFunctionalConsequenceId': ['SO_0001583'] * 5 + ['SO_0001587'],
        ...     'clinicalSignificances': [['pathogenic'], ['likely pathogenic'], ['benign'], ['benign'], ['pathogenic'],
        ...                               ['pathogenic']],
        ...     'confidence': ['criteria provided, single submitter'] * 6,
        ... })
        >>> clinvar_missense_labels(eva).sort('variantId').collect()
        shape: (2, 3)
        ┌───────────┬──────────┬───────┐
        │ variantId ┆ targetId ┆ label │
        │ ---       ┆ ---      ┆ ---   │
        │ str       ┆ str      ┆ i32   │
        ╞═══════════╪══════════╪═══════╡
        │ v1        ┆ g        ┆ 1     │
        │ v2        ┆ g        ┆ 0     │
        └───────────┴──────────┴───────┘
    """
    if min_review not in ('single', 'multiple'):
        raise ValueError(f"min_review must be 'single' or 'multiple', got {min_review!r}")
    confidence = pl.col('confidence')
    reviewed = confidence.is_in(REVIEWED_CONFIDENCE) | (
        confidence.str.starts_with('criteria provided') & ~confidence.str.contains('conflicting')
    )
    if min_review == 'multiple':
        reviewed &= ~confidence.str.contains('single submitter')
    significance = pl.col('clinicalSignificances')
    return (
        evidence_eva
        .filter(pl.col('variantFunctionalConsequenceId') == MISSENSE)
        .filter(pl.col('variantId').is_not_null() & reviewed)
        .with_columns(
            pathogenic=significance.list.eval(pl.element().is_in(PATHOGENIC_LABELS)).list.any(),
            benign=significance.list.eval(pl.element().is_in(BENIGN_LABELS)).list.any(),
        )
        .filter(pl.col('pathogenic') | pl.col('benign'))
        .group_by('variantId')
        .agg(pl.col('targetId').first(), pl.col('pathogenic').any(), pl.col('benign').any())
        .filter(pl.col('pathogenic') != pl.col('benign'))
        .select('variantId', 'targetId', pl.col('pathogenic').cast(pl.Int32).alias('label'))
    )


def variant_scores(variant: pl.LazyFrame, variants: pl.LazyFrame) -> pl.LazyFrame:
    """The GPN-Star LLR and, per gene, the AlphaMissense pathogenicity of the `variants` (`variantId`, `targetId`).

    `llr` is null for a variant without a GPN-Star effect (indels, or no score at the position);
    `am_pathogenicity` is null without an AlphaMissense effect for that gene.

    Examples:
        >>> effect = lambda method, score, target=None: {'method': method, 'score': score, 'targetId': target}
        >>> index = pl.LazyFrame({
        ...     'variantId': ['v1', 'v2', 'v3'],
        ...     'variantEffect': [[effect('GPN-Star', -3.0), effect('AlphaMissense', 0.9, 'g')],
        ...                       [effect('AlphaMissense', 0.1, 'h')], None],
        ... })
        >>> wanted = pl.LazyFrame({'variantId': ['v1', 'v2', 'v3'], 'targetId': ['g', 'g', 'g']})
        >>> variant_scores(index, wanted).sort('variantId').collect()
        shape: (3, 4)
        ┌───────────┬──────────┬──────┬──────────────────┐
        │ variantId ┆ targetId ┆ llr  ┆ am_pathogenicity │
        │ ---       ┆ ---      ┆ ---  ┆ ---              │
        │ str       ┆ str      ┆ f64  ┆ f64              │
        ╞═══════════╪══════════╪══════╪══════════════════╡
        │ v1        ┆ g        ┆ -3.0 ┆ 0.9              │
        │ v2        ┆ g        ┆ null ┆ null             │
        │ v3        ┆ g        ┆ null ┆ null             │
        └───────────┴──────────┴──────┴──────────────────┘
    """
    effects = (
        variant
        .select('variantId', 'variantEffect')
        .join(variants.select('variantId').unique(), on='variantId', how='semi')
        .explode('variantEffect')
        .filter(pl.col('variantEffect').is_not_null())
        .select(
            'variantId',
            pl.col('variantEffect').struct.field('method').alias('method'),
            pl.col('variantEffect').struct.field('score').cast(pl.Float64).alias('score'),
            pl.col('variantEffect').struct.field('targetId').alias('targetId'),
        )
    )
    gpn_star = (
        effects.filter(pl.col('method') == 'GPN-Star').group_by('variantId').agg(pl.col('score').first().alias('llr'))
    )
    alphamissense = (
        effects
        .filter(pl.col('method') == 'AlphaMissense')
        .group_by('variantId', 'targetId')
        .agg(pl.col('score').first().alias('am_pathogenicity'))
    )
    return variants.join(gpn_star, on='variantId', how='left').join(
        alphamissense, on=['variantId', 'targetId'], how='left'
    )


def fit_logistic(x: np.ndarray, y: np.ndarray, *, balanced: bool = False) -> tuple[float, float]:
    """Maximum-likelihood `a, b` of `P(y = 1 | x) = sigmoid(a + b * x)`.

    `balanced` weights each class to the same total, which fits the curve at a 50:50 prevalence
    instead of the sample's. The fit runs on standardised `x` and the coefficients are mapped back.

    Examples:
        >>> rng = np.random.default_rng(0)
        >>> x = rng.normal(size=20000)
        >>> y = (rng.uniform(size=x.size) < 1 / (1 + np.exp(-(0.5 - 2.0 * x)))).astype(int)
        >>> a, b = fit_logistic(x, y)
        >>> bool(abs(a - 0.5) < 0.1 and abs(b + 2.0) < 0.1)
        True
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or x.shape != y.shape:
        raise ValueError('x and y must be one-dimensional and the same length')
    if not (np.any(y == 1) and np.any(y == 0)):
        raise ValueError('both classes must be present')
    mean, scale = x.mean(), x.std() or 1.0
    z = (x - mean) / scale
    weights = np.ones_like(y)
    if balanced:
        weights = np.where(y == 1, 0.5 / y.mean(), 0.5 / (1 - y.mean()))

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = theta[0] + theta[1] * z
        p = 1 / (1 + np.exp(-eta))
        loss = -np.sum(weights * (y * np.log(p + 1e-300) + (1 - y) * np.log(1 - p + 1e-300)))
        residual = weights * (p - y)
        return loss, np.array([residual.sum(), (residual * z).sum()])

    result = minimize(objective, x0=np.zeros(2), jac=True, method='L-BFGS-B')
    if not result.success:
        raise RuntimeError(f'logistic fit did not converge: {result.message}')
    intercept, slope = result.x
    return float(intercept - slope * mean / scale), float(slope / scale)


def auroc(score: np.ndarray, y: np.ndarray) -> float:
    """Area under the ROC curve of `score` for `y == 1`, by the rank-sum formula.

    Examples:
        >>> round(auroc(np.array([0.1, 0.4, 0.35, 0.8]), np.array([0, 0, 1, 1])), 2)
        0.75
    """
    y = np.asarray(y, dtype=bool)
    ranks = rankdata(np.asarray(score, dtype=float))
    n_pos, n_neg = y.sum(), (~y).sum()
    if n_pos == 0 or n_neg == 0:
        raise ValueError('both classes must be present')
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


@dataclass(frozen=True)
class Calibration:
    """The fitted transform and the diagnostics of the fit."""

    a: float
    b: float
    balanced: bool
    min_review: str
    n_pathogenic: int
    n_benign: int
    n_without_llr: int
    """Labelled missense variants without a GPN-Star score, excluded from the fit."""
    auroc_gpn_star: float
    n_with_alphamissense: int
    auroc_gpn_star_on_alphamissense_subset: float | None
    auroc_alphamissense: float | None
    benign_score_quantiles: dict[str, list[float]]
    """5/25/50/75/95% of `benign_score` and of `1 - am_pathogenicity`, per class."""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def calibrate(
    variant: pl.LazyFrame, evidence_eva: pl.LazyFrame, *, min_review: str = 'single', balanced: bool = False
) -> Calibration:
    """Fit `a, b` on the ClinVar missense variants of a release and report how the transform behaves.

    Args:
        variant: the variant index (`variantId`, `variantEffect`).
        evidence_eva: the EVA evidence (`variantId`, `targetId`, `variantFunctionalConsequenceId`,
            `clinicalSignificances`, `confidence`).
        min_review: see `clinvar_missense_labels`.
        balanced: see `fit_logistic`.
    """
    labels = clinvar_missense_labels(evidence_eva, min_review=min_review)
    scored = variant_scores(variant, labels).collect()
    fit = scored.filter(pl.col('llr').is_not_null())
    x, y = fit['llr'].to_numpy(), fit['label'].to_numpy()
    a, b = fit_logistic(x, y, balanced=balanced)
    fit = fit.with_columns(
        benign_score(pl.col('llr'), a, b).alias('gpn_benign'),
        (1.0 - pl.col('am_pathogenicity')).alias('am_benign'),
    )
    with_am = fit.filter(pl.col('am_pathogenicity').is_not_null())
    quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]

    def summary(frame: pl.DataFrame, column: str, label: int) -> list[float]:
        values = frame.filter(pl.col('label') == label)[column].drop_nulls()
        return [round(float(values.quantile(q)), 4) for q in quantiles] if len(values) else []

    return Calibration(
        a=a,
        b=b,
        balanced=balanced,
        min_review=min_review,
        n_pathogenic=int((y == 1).sum()),
        n_benign=int((y == 0).sum()),
        n_without_llr=int(scored['llr'].null_count()),
        auroc_gpn_star=round(auroc(-x, y), 4),
        n_with_alphamissense=len(with_am),
        auroc_gpn_star_on_alphamissense_subset=(
            round(auroc(-with_am['llr'].to_numpy(), with_am['label'].to_numpy()), 4) if len(with_am) else None
        ),
        auroc_alphamissense=(
            round(auroc(with_am['am_pathogenicity'].to_numpy(), with_am['label'].to_numpy()), 4)
            if len(with_am)
            else None
        ),
        benign_score_quantiles={
            'gpn_star_pathogenic': summary(fit, 'gpn_benign', 1),
            'gpn_star_benign': summary(fit, 'gpn_benign', 0),
            'alphamissense_pathogenic': summary(with_am, 'am_benign', 1),
            'alphamissense_benign': summary(with_am, 'am_benign', 0),
        },
    )
