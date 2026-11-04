"""LD score regression for one trait (heritability) and for two (genetic covariance).

A port of `Hsq` and `Gencov` from `ldscore/regressions.py` in github.com/bulik/ldsc
(Bulik-Sullivan et al. 2015, doi:10.1038/ng.3406), for a single LD score annotation. Where this
module differs from a textbook reading of the method, it follows the LDSC code, so that estimates
match the reference implementation to floating point precision.

Both regressions fit `y ~ intercept + (N / M) * ld * estimate`: chi-square on LD score for
heritability, the product of two traits' z-scores for their genetic covariance. Weights are
updated twice from the running estimate (`jackknife.reweighted_least_squares`), and with a
two-step cut-off the intercept is first fitted on the SNPs below it and then held fixed.

Arrays are 1-D per SNP on the way in and 2-D inside, as LDSC keeps them.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from pts.transformers.genetic_correlation.ldsc import jackknife

_LAMBDA_GC_MEDIAN_CHISQ = 0.4549
"""Median of a chi-square distribution with one degree of freedom."""


@dataclass(frozen=True)
class Fit:
    """One LD score regression."""

    total: float
    """The heritability or genetic covariance: slope times `m`."""
    total_se: float
    intercept: float
    intercept_se: float
    """NaN when the intercept was fixed."""
    total_delete_values: np.ndarray
    """`total` with each jackknife block left out, shape (n_blocks,)."""


@dataclass(frozen=True)
class Heritability(Fit):
    """A heritability regression, with the chi-square summaries LDSC reports beside it."""

    mean_chisq: float
    lambda_gc: float


Weights = Callable[[np.ndarray, np.ndarray, np.ndarray | None, float, float], np.ndarray]
"""Inverse-variance weights from (ld, w_ld, mask, estimate, intercept): `ld` and `w_ld` are per
SNP, already cut to the SNPs `mask` selects (None for all of them)."""


def heritability_weights(
    ld: np.ndarray, w_ld: np.ndarray, n: np.ndarray, m: float, h2: float, intercept: float
) -> np.ndarray:
    """LDSC's weights for the chi-square regression: the inverse of its approximate variance."""
    h2 = min(max(h2, 0.0), 1.0)
    ld = np.fmax(ld, 1.0)
    w_ld = np.fmax(w_ld, 1.0)
    heteroscedasticity = 1.0 / (2 * np.square(intercept + h2 * n * ld / m))
    return heteroscedasticity / w_ld


def covariance_weights(
    ld: np.ndarray,
    w_ld: np.ndarray,
    n1: np.ndarray,
    n2: np.ndarray,
    m: float,
    h2_1: float,
    h2_2: float,
    intercept_1: float,
    intercept_2: float,
    rho_g: float,
    intercept: float,
) -> np.ndarray:
    """LDSC's weights for the z1 * z2 regression: the inverse of its approximate variance."""
    h2_1 = min(max(h2_1, 0.0), 1.0)
    h2_2 = min(max(h2_2, 0.0), 1.0)
    rho_g = min(max(rho_g, -1.0), 1.0)
    ld = np.fmax(ld, 1.0)
    w_ld = np.fmax(w_ld, 1.0)
    a = n1 * h2_1 * ld / m + intercept_1
    b = n2 * h2_2 * ld / m + intercept_2
    c = np.sqrt(n1 * n2) * rho_g * ld / m + intercept
    heteroscedasticity = 1.0 / (a * b + np.square(c))
    return heteroscedasticity / w_ld


def regress(
    y: np.ndarray,
    ld: np.ndarray,
    w_ld: np.ndarray,
    n: np.ndarray,
    m: float,
    weights: Weights,
    *,
    null_intercept: float,
    n_blocks: int,
    intercept: float | None,
    step1: np.ndarray | None,
) -> Fit:
    """Fit `y ~ intercept + (n / m) * ld * total` with LDSC's reweighting and jackknife.

    Args:
        y: response per SNP, shape (n_snp,).
        ld: LD score per SNP.
        w_ld: LD score per SNP for the weights, over the regression SNPs.
        n: sample size per SNP.
        m: number of SNPs the LD scores were computed over.
        weights: the regression's weight function.
        null_intercept: the intercept the first weights assume, 1 for chi-square and 0 for z1 * z2.
        n_blocks: jackknife blocks.
        intercept: fix the intercept to this value; None estimates it.
        step1: with a free intercept, the SNPs to fit it on first (the two-step estimator).

    Returns:
        The fit.

    Raises:
        ValueError: for a two-step fit with a fixed intercept.
    """
    if step1 is not None and intercept is not None:
        raise ValueError('the two-step estimator needs a free intercept')
    n_snp = len(y)
    column = (n_snp, 1)
    y = y.reshape(column).astype(float)
    ld = ld.reshape(column).astype(float)
    w_ld = w_ld.reshape(column).astype(float)
    n = n.reshape(column).astype(float)

    first_intercept = null_intercept if intercept is None else intercept
    aggregate = m * (y.mean() - first_intercept) / (ld * n).mean()
    initial_w = weights(ld, w_ld, None, aggregate, first_intercept)
    n_bar = float(n.mean())
    x = n * ld / n_bar

    def update(
        ld_for_weights: np.ndarray, w: np.ndarray, mask: np.ndarray | None, fixed: float | None
    ) -> Callable[[np.ndarray], np.ndarray]:
        def new_weights(coefficients: np.ndarray) -> np.ndarray:
            estimate = m * coefficients[0, 0] / n_bar
            current = coefficients[1, 0] if fixed is None else fixed
            return weights(ld_for_weights, w, mask, estimate, current)

        return new_weights

    boundaries = jackknife.separators(n_snp, n_blocks)
    if intercept is not None:
        fit = jackknife.reweighted_least_squares(
            x, y - intercept, update(ld, w_ld, None, intercept), initial_w, boundaries
        )
        return Fit(
            total=m * float(fit.est[0, 0]) / n_bar,
            total_se=m * float(fit.jackknife_se[0, 0]) / n_bar,
            intercept=intercept,
            intercept_se=math.nan,
            total_delete_values=fit.delete_values[:, 0] * m / n_bar,
        )

    with_intercept = np.hstack([x, np.ones(column)])
    if step1 is None:
        fit = jackknife.reweighted_least_squares(with_intercept, y, update(ld, w_ld, None, None), initial_w, boundaries)
    else:
        # LDSC passes the step-one design, N * ld / Nbar, where the weights expect raw LD scores;
        # kept so the estimates match it
        mask = step1.reshape(n_snp)
        step1_fit = jackknife.reweighted_least_squares(
            with_intercept[mask],
            y[mask],
            update(x[mask], w_ld[mask], mask, None),
            initial_w[mask],
            jackknife.separators(int(mask.sum()), n_blocks),
        )
        step1_intercept = float(step1_fit.est[0, 1])
        step2_fit = jackknife.reweighted_least_squares(
            x,
            y - step1_intercept,
            update(ld, w_ld, None, step1_intercept),
            initial_w,
            jackknife.unmask_separators(step1_fit.separators, mask),
        )
        # the slope's delete values are corrected for the intercept's, by how much the slope
        # moves per unit of intercept under the initial weights
        c = float((initial_w * x).sum() / (initial_w * np.square(x)).sum())
        est = np.array([[step2_fit.est[0, 0], step1_intercept]])
        delete_values = np.column_stack([
            step2_fit.delete_values[:, 0] - c * (step1_fit.delete_values[:, 1] - step1_intercept),
            step1_fit.delete_values[:, 1],
        ])
        jackknife_est, se, cov = jackknife.summarise(jackknife.pseudovalues(delete_values, est))
        fit = jackknife.Jackknife(est, delete_values, jackknife_est, se, cov, step2_fit.separators)
    return Fit(
        total=m * float(fit.est[0, 0]) / n_bar,
        total_se=m * float(fit.jackknife_se[0, 0]) / n_bar,
        intercept=float(fit.est[0, 1]),
        intercept_se=float(fit.jackknife_se[0, 1]),
        total_delete_values=fit.delete_values[:, 0] * m / n_bar,
    )


def heritability(
    chisq: np.ndarray,
    ld: np.ndarray,
    w_ld: np.ndarray,
    n: np.ndarray,
    m: float,
    *,
    n_blocks: int = 200,
    intercept: float | None = None,
    twostep: float | None = None,
) -> Heritability:
    """SNP heritability of one trait from its chi-square statistics.

    Args:
        chisq: squared z-score per SNP, shape (n_snp,).
        ld: LD score per SNP.
        w_ld: LD score per SNP for the weights.
        n: sample size per SNP.
        m: number of SNPs the LD scores were computed over.
        n_blocks: jackknife blocks.
        intercept: fix the intercept; None estimates it.
        twostep: fit the intercept first on SNPs with chi-square below this; None fits in one step.

    Returns:
        The heritability fit.
    """
    chisq = np.asarray(chisq, dtype=float)
    n = np.asarray(n, dtype=float)

    def weights(
        ld_: np.ndarray, w_ld_: np.ndarray, mask: np.ndarray | None, h2: float, intercept_: float
    ) -> np.ndarray:
        n_ = n if mask is None else n[mask]
        return heritability_weights(ld_, w_ld_, n_.reshape(ld_.shape), m, h2, intercept_)

    fit = regress(
        chisq,
        ld,
        w_ld,
        n,
        m,
        weights,
        null_intercept=1.0,
        n_blocks=n_blocks,
        intercept=intercept,
        step1=None if twostep is None else chisq < twostep,
    )
    return Heritability(
        **vars(fit),
        mean_chisq=float(chisq.mean()),
        lambda_gc=float(np.median(chisq) / _LAMBDA_GC_MEDIAN_CHISQ),
    )


def genetic_covariance(
    z1: np.ndarray,
    z2: np.ndarray,
    ld: np.ndarray,
    w_ld: np.ndarray,
    n1: np.ndarray,
    n2: np.ndarray,
    m: float,
    first: Fit,
    second: Fit,
    *,
    n_blocks: int = 200,
    intercept: float | None = None,
    twostep: float | None = None,
) -> Fit:
    """Genetic covariance of two traits from the product of their z-scores.

    Args:
        z1: z-score per SNP in the first trait, shape (n_snp,).
        z2: z-score per SNP in the second trait.
        ld: LD score per SNP.
        w_ld: LD score per SNP for the weights.
        n1: sample size per SNP in the first trait.
        n2: sample size per SNP in the second trait.
        m: number of SNPs the LD scores were computed over.
        first: the first trait's heritability fit, for the weights.
        second: the second trait's heritability fit, for the weights.
        n_blocks: jackknife blocks.
        intercept: fix the intercept; None estimates it.
        twostep: fit the intercept first on SNPs with both chi-squares below this.

    Returns:
        The genetic covariance fit.
    """
    z1 = np.asarray(z1, dtype=float)
    z2 = np.asarray(z2, dtype=float)
    n1 = np.asarray(n1, dtype=float)
    n2 = np.asarray(n2, dtype=float)

    def weights(
        ld_: np.ndarray, w_ld_: np.ndarray, mask: np.ndarray | None, rho_g: float, intercept_: float
    ) -> np.ndarray:
        shape = ld_.shape
        n1_, n2_ = (n1, n2) if mask is None else (n1[mask], n2[mask])
        return covariance_weights(
            ld_,
            w_ld_,
            n1_.reshape(shape),
            n2_.reshape(shape),
            m,
            first.total,
            second.total,
            first.intercept,
            second.intercept,
            rho_g,
            intercept_,
        )

    return regress(
        z1 * z2,
        ld,
        w_ld,
        np.sqrt(n1 * n2),
        m,
        weights,
        null_intercept=0.0,
        n_blocks=n_blocks,
        intercept=intercept,
        step1=None if twostep is None else (np.square(z1) < twostep) & (np.square(z2) < twostep),
    )
