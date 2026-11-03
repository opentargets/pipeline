"""Fit one regression per study and biosample, all at once, with region-clustered errors.

Each regression asks whether a study's L2G scores rise with a biosample's specificity, after
allowing for gene length and density. A release has millions of such pairs, so instead of
fitting them one by one, the length and density adjustment is done once per biosample and
reused for every study (the Frisch-Waugh-Lovell theorem guarantees the same answer). The
results match statsmodels exactly, which `tests/test_enrichment_regression.py` checks.

Standard errors treat genes in the same 1 Mb region as one piece of evidence, since nearby
tissue-specific genes can all be picked up by a single association.

Arrays are genes x columns: outcomes are genes x units (a unit is a study, or a pooled group
of studies), predictors genes x biosamples, and results units x biosamples.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

_CLUSTER_BATCH = 64
"""Units per pass when summing scores by cluster. Each pass holds a clusters x batch x biosamples
array: about 80 MB for 3,000 clusters and 50 biosamples."""

_NO_VARIANCE = 1e-10
"""Residual sum of squares, relative to the predictor's own, below which it counts as zero."""

StandardErrors = Literal['region', 'gene', 'classical']


def residualise(covariates: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Project the columns of `matrix` off the column space of `covariates`.

    Args:
        covariates: genes x p, including the intercept column.
        matrix: genes x k.

    Returns:
        genes x k residuals.
    """
    q, _ = np.linalg.qr(covariates)
    return matrix - q @ (q.T @ matrix)


@dataclass(frozen=True)
class Clusters:
    """A grouping of genes, kept as a sort order so a cluster sum is one `reduceat`."""

    codes: np.ndarray
    """Index of each gene's cluster, in the order `sum` returns clusters."""
    order: np.ndarray
    """Gene indices sorted so that each cluster's genes are contiguous."""
    starts: np.ndarray
    """Offset into `order` at which each cluster begins."""

    @classmethod
    def from_labels(cls, labels: np.ndarray) -> Clusters:
        """Group genes that share a label.

        Args:
            labels: one label per gene, of any sortable type.

        Returns:
            The clusters.
        """
        codes = np.unique(labels, return_inverse=True)[1].ravel()
        order = np.argsort(codes, kind='stable')
        boundaries = np.diff(codes[order]) != 0
        starts = np.flatnonzero(np.concatenate([[True], boundaries]))
        return cls(codes=codes, order=order, starts=starts)

    @property
    def count(self) -> int:
        """Number of clusters."""
        return len(self.starts)

    def sum(self, matrix: np.ndarray) -> np.ndarray:
        """Add up the rows of `matrix` within each cluster.

        Args:
            matrix: genes x ..., summed along the first axis.

        Returns:
            clusters x ... sums, in the order of `codes`.
        """
        return np.add.reduceat(matrix[self.order], self.starts, axis=0)


def region_labels(chromosome: np.ndarray, position: np.ndarray, window: int) -> np.ndarray:
    """Label each gene with the fixed genomic window its position falls in.

    Windows are fixed, not sliding: `chromosome:floor(position / window)`.

    Args:
        chromosome: chromosome per gene.
        position: position per gene, normally the TSS.
        window: window size in base pairs.

    Returns:
        A string label per gene.
    """
    bins = (np.asarray(position, dtype=np.int64) // window).astype(str)
    return np.char.add(np.char.add(np.asarray(chromosome, dtype=str), ':'), bins)


@dataclass(frozen=True)
class Fit:
    """Slopes and their standard errors, units x biosamples.

    Both are NaN where a predictor has no variance left after the covariates.
    """

    beta: np.ndarray
    standard_error: np.ndarray


class Design:
    """A fixed set of covariates, predictors and clusters, to fit many outcomes against.

    Everything that does not depend on the outcome is computed here, once per panel, and
    reused for every unit passed to `fit`.
    """

    def __init__(
        self,
        covariates: np.ndarray,
        predictors: np.ndarray,
        standard_errors: StandardErrors,
        clusters: Clusters | None = None,
    ) -> None:
        """Prepare the design.

        Args:
            covariates: genes x p, including the intercept column.
            predictors: genes x biosamples.
            standard_errors: `region` for cluster-robust errors over `clusters`; `gene` for one
                cluster per gene (HC0); `classical` for one common noise variance.
            clusters: the gene clusters; required for, and only used by, `region`.

        Raises:
            ValueError: if `standard_errors` is `region` and no clusters are given.
        """
        if standard_errors == 'region' and clusters is None:
            raise ValueError('region standard errors need clusters')
        self.standard_errors = standard_errors
        self._q, _ = np.linalg.qr(covariates)
        # Covariate adjustment, once per biosample: keep the part of each biosample's
        # specificity that gene length and density do not explain. Frisch-Waugh-Lovell: a
        # regression on this residual gives exactly the full model's slope.
        self._xr = residualise(covariates, predictors)
        # a predictor the covariates explain leaves rounding noise, not zero; treat it as none
        sxx = (self._xr**2).sum(0)
        self._sxx = np.where(sxx > _NO_VARIANCE * (predictors**2).sum(0), sxx, 0.0)
        self._df = covariates.shape[0] - covariates.shape[1] - 1
        self._clusters = clusters if standard_errors == 'region' else None
        if self._clusters is not None:
            # per-region totals that `_noise_by_region` reuses for every unit
            self._xx = self._clusters.sum(self._xr**2)
            self._xq = self._clusters.sum(self._xr[:, :, None] * self._q[:, None, :])

    def fit(self, y: np.ndarray) -> Fit:
        """Fit every outcome column against every predictor column.

        Args:
            y: genes x units outcomes, not residualised.

        Returns:
            The fit, units x biosamples.
        """
        qy = self._q.T @ y
        # The slope can use the raw outcome: the adjusted specificity is already independent of
        # the covariates, so adjusting y too would not change it. One matrix product per batch.
        with np.errstate(divide='ignore', invalid='ignore'):
            beta = np.where(self._sxx > 0, (y.T @ self._xr) / self._sxx, np.nan)
        if self.standard_errors == 'classical':
            # one noise variance for every gene: the textbook OLS standard error
            syy = (y**2).sum(0) - (qy**2).sum(0)
            rss = np.maximum(syy[:, None] - beta**2 * self._sxx, 0.0)
            with np.errstate(divide='ignore', invalid='ignore'):
                return Fit(beta, np.where(np.isnan(beta), np.nan, np.sqrt(rss / self._df / self._sxx)))
        # Sandwich standard errors: built from the residuals themselves rather than assuming
        # every gene is equally noisy, which L2G (mostly 0, a few large scores) is not.
        beta0 = np.nan_to_num(beta)
        noise = self._noise_by_region(y, qy, beta0) if self._clusters is not None else self._noise_by_gene(y, qy, beta0)
        with np.errstate(divide='ignore', invalid='ignore'):
            return Fit(beta, np.where(np.isnan(beta), np.nan, np.sqrt(np.maximum(noise, 0.0)) / self._sxx))

    def _noise_by_gene(self, y: np.ndarray, qy: np.ndarray, beta0: np.ndarray) -> np.ndarray:
        """How much the slope could wobble from noise, letting each gene be as noisy as it is (HC0).

        Computes, per unit and biosample, sum over genes of (x~_g * e_g)^2: each gene's adjusted
        specificity times its residual, squared. This is the middle term of the sandwich variance,
        so the standard error is sqrt(this) / sum(x~^2).
        """
        yr = y - self._q @ qy
        x2 = self._xr**2
        return (yr**2).T @ x2 - 2 * beta0 * (yr.T @ (x2 * self._xr)) + beta0**2 * (x2**2).sum(0)

    def _noise_by_region(self, y: np.ndarray, qy: np.ndarray, beta0: np.ndarray) -> np.ndarray:
        """As `_noise_by_gene`, but genes in one region pool their residuals and count as one.

        Nearby tissue-specific genes can all be picked up by one association, so a per-gene error
        would count that one signal several times and overstate the evidence.

        Computes, per unit and biosample, sum over regions of (sum over the region's genes of
        x~_g * e_g)^2: the products are added up within each region first, then squared.
        """
        clusters = self._clusters
        assert clusters is not None  # only called with clusters
        n_clusters, n_biosamples, p = self._xq.shape
        xq = self._xq.reshape(n_clusters * n_biosamples, p)
        noise = np.empty_like(beta0)
        for start in range(0, y.shape[1], _CLUSTER_BATCH):
            stop = min(start + _CLUSTER_BATCH, y.shape[1])
            genes, units = np.nonzero(y[:, start:stop])
            scores = np.zeros((n_clusters, stop - start, n_biosamples))
            # A unit's L2G is almost all zeros, so this sums over only its few scored genes, never
            # all ~19k: the reason a release-sized run takes minutes rather than hours.
            np.add.at(scores, (clusters.codes[genes], units), y[genes, start + units][:, None] * self._xr[genes])
            # then subtract what the covariates and the slope explain, using the per-region totals
            dense = (xq @ qy[:, start:stop]).reshape(n_clusters, n_biosamples, stop - start)
            scores -= dense.transpose(0, 2, 1)
            scores -= beta0[None, start:stop, :] * self._xx[:, None, :]
            noise[start:stop] = (scores**2).sum(0)
        return noise


_ERFC = np.vectorize(math.erfc, otypes=[float])


def upper_tail_p(z: np.ndarray) -> np.ndarray:
    """One-sided p-value `P(Z >= z)` under a standard normal, keeping NaN as NaN.

    Uses `math.erfc` rather than `1 - cdf`, which keeps precision far into the tail, and
    avoids a scipy dependency that pts otherwise only has through optional extras.

    Args:
        z: z-statistics of any shape.

    Returns:
        p-values of the same shape.
    """
    out = np.full(z.shape, np.nan)
    finite = np.isfinite(z)
    out[finite] = 0.5 * _ERFC(z[finite] / math.sqrt(2.0))
    return out


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q-values along each row, ignoring NaN.

    Each row is a separate family: one unit tested across a panel's biosamples. The number of
    tests in a row is its count of non-NaN p-values.

    Args:
        p: units x biosamples p-values.

    Returns:
        q-values of the same shape, NaN where `p` is NaN.
    """
    m = np.isfinite(p).sum(1, keepdims=True)
    order = np.argsort(np.where(np.isfinite(p), p, np.inf), axis=1)
    ranked = np.take_along_axis(p, order, axis=1)
    rank = np.arange(1, p.shape[1] + 1)[None, :]
    with np.errstate(invalid='ignore'):
        q = ranked * m / rank
    # running minimum from the largest p down; NaN sorts last and stays out of it
    q = np.where(np.isfinite(q), q, np.inf)
    q = np.minimum.accumulate(q[:, ::-1], axis=1)[:, ::-1]
    q = np.minimum(q, 1.0)
    out = np.empty_like(q)
    np.put_along_axis(out, order, q, axis=1)
    return np.where(np.isfinite(p), out, np.nan)


def bonferroni(p: np.ndarray) -> np.ndarray:
    """Bonferroni-adjusted p-values along each row, ignoring NaN.

    Args:
        p: units x biosamples p-values.

    Returns:
        adjusted p-values of the same shape, capped at 1.
    """
    m = np.isfinite(p).sum(1, keepdims=True)
    return np.minimum(p * m, 1.0)
