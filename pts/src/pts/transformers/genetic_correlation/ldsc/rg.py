"""Genetic correlation of two traits, as `ldsc --rg` computes it.

rg is the genetic covariance over the geometric mean of the two heritabilities. Its standard
error is a block jackknife of that ratio (`jackknife.ratio`), which carries the dependence
between the three regressions; LDSC reports no rg when either heritability is not positive.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pts.transformers.genetic_correlation.ldsc import jackknife, regression


@dataclass(frozen=True)
class GeneticCorrelation:
    """One pair's estimates. Every field but `n_snps` is NaN where it is undefined."""

    rg: float
    rg_se: float
    genetic_covariance: float
    genetic_covariance_se: float
    intercept: float
    """The genetic covariance intercept; near zero without sample overlap."""
    intercept_se: float
    h2_1: float
    h2_1_se: float
    h2_1_intercept: float
    h2_2: float
    h2_2_se: float
    h2_2_intercept: float
    n_snps: int


def genetic_correlation(
    z1: np.ndarray,
    z2: np.ndarray,
    n1: np.ndarray,
    n2: np.ndarray,
    ld: np.ndarray,
    w_ld: np.ndarray,
    m: float,
    *,
    n_blocks: int = 200,
    twostep: float | None = 30.0,
    intercept: float | None = None,
) -> GeneticCorrelation:
    """Estimate the genetic correlation of two traits over their shared SNPs.

    Args:
        z1: z-score per SNP in the first trait, shape (n_snp,).
        z2: z-score per SNP in the second trait.
        n1: sample size per SNP in the first trait.
        n2: sample size per SNP in the second trait.
        ld: LD score per SNP.
        w_ld: LD score per SNP for the weights, over the regression SNPs.
        m: number of SNPs the LD scores were computed over.
        n_blocks: jackknife blocks; needs at least that many SNPs.
        twostep: the two-step cut-off on chi-square; None fits every regression in one step.
        intercept: fix the genetic covariance intercept; None estimates it.

    Returns:
        The estimates.

    Raises:
        ValueError: if the arrays are not all 1-D of one length, or there are fewer SNPs than blocks.
    """
    arrays = [np.asarray(a, dtype=float) for a in (z1, z2, n1, n2, ld, w_ld)]
    shape = arrays[0].shape
    if len(shape) != 1 or any(a.shape != shape for a in arrays):
        raise ValueError('z1, z2, n1, n2, ld and w_ld must be 1-D arrays of one length')
    n_snps = shape[0]
    if n_snps < n_blocks:
        raise ValueError(f'{n_snps} SNPs cannot fill {n_blocks} jackknife blocks')
    z1, z2, n1, n2, ld, w_ld = arrays

    first = regression.heritability(np.square(z1), ld, w_ld, n1, m, n_blocks=n_blocks, twostep=twostep)
    second = regression.heritability(np.square(z2), ld, w_ld, n2, m, n_blocks=n_blocks, twostep=twostep)
    covariance = regression.genetic_covariance(
        z1, z2, ld, w_ld, n1, n2, m, first, second, n_blocks=n_blocks, intercept=intercept, twostep=twostep
    )

    if first.total <= 0 or second.total <= 0:
        rg, rg_se = math.nan, math.nan
    else:
        rg = covariance.total / math.sqrt(first.total * second.total)
        with np.errstate(invalid='ignore'):
            denominator = np.sqrt(first.total_delete_values * second.total_delete_values)
        _, rg_se = jackknife.ratio(rg, covariance.total_delete_values, denominator)

    return GeneticCorrelation(
        rg=rg,
        rg_se=rg_se,
        genetic_covariance=covariance.total,
        genetic_covariance_se=covariance.total_se,
        intercept=covariance.intercept,
        intercept_se=covariance.intercept_se,
        h2_1=first.total,
        h2_1_se=first.total_se,
        h2_1_intercept=first.intercept,
        h2_2=second.total,
        h2_2_se=second.total_se,
        h2_2_intercept=second.intercept,
        n_snps=n_snps,
    )
