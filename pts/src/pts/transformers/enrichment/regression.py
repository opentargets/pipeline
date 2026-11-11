"""Regress every unit's L2G scores on every biosample's specificity, adjusting for covariates.

`enrichment` is the entry point: one ordinary least squares slope, standard error and one-sided
p-value per unit and biosample, with cluster-robust errors over genomic regions by default.
`Design` holds what does not depend on the outcome, so it is computed once per panel.

Arrays are genes x columns: outcomes are genes x units (a unit is a study, or a pooled group
of studies), predictors genes x biosamples, and results units x biosamples.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

import numpy as np

DEFAULT_MAX_BYTES = 128 * 2**20
"""Default size of the largest array a batch of units holds."""

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
        *,
        max_bytes: int = DEFAULT_MAX_BYTES,
    ) -> None:
        """Prepare the design.

        Args:
            covariates: genes x p, including the intercept column.
            predictors: genes x biosamples.
            standard_errors: `region` for cluster-robust errors over `clusters`; `gene` for one
                cluster per gene (HC0); `classical` for one common noise variance.
            clusters: the gene clusters; required for, and only used by, `region`.
            max_bytes: size of the largest array `fit` holds when summing by region.

        Raises:
            ValueError: if `standard_errors` is `region` and no clusters are given.
        """
        if standard_errors == 'region' and clusters is None:
            raise ValueError('region standard errors need clusters')
        self.standard_errors = standard_errors
        self._max_bytes = max_bytes
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
        batch = max(1, self._max_bytes // (8 * n_clusters * n_biosamples))
        for start in range(0, y.shape[1], batch):
            stop = min(start + batch, y.shape[1])
            genes, units = np.nonzero(y[:, start:stop])
            scores = np.zeros((n_clusters, stop - start, n_biosamples))
            # sum over each unit's scored genes only; its other scores are zero
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

    Computed with `erfc`, so it stays above zero far into the tail.

    Args:
        z: z-statistics of any shape.

    Returns:
        p-values of the same shape.
    """
    out = np.full(z.shape, np.nan)
    finite = np.isfinite(z)
    out[finite] = 0.5 * _ERFC(z[finite] / math.sqrt(2.0))
    return out


def standardise(values: np.ndarray) -> np.ndarray:
    """Centre each column and scale it to unit variance; a constant column is only centred."""
    spread = values.std(0)
    return (values - values.mean(0)) / np.where(spread > 0, spread, 1.0)


@dataclass(frozen=True)
class Outcomes:
    """A genes x units outcome matrix, given by its nonzero entries."""

    genes: np.ndarray
    units: np.ndarray
    values: np.ndarray
    n_units: int


@dataclass(frozen=True)
class Enrichment:
    """The results for units `start` to `start + len(beta)`, units x biosamples, NaN where undefined."""

    start: int
    beta: np.ndarray
    standard_error: np.ndarray
    p_value: np.ndarray


def enrichment(
    covariates: np.ndarray,
    predictors: np.ndarray,
    outcomes: Outcomes,
    standard_errors: StandardErrors,
    regions: np.ndarray | None = None,
    *,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> Iterator[Enrichment]:
    """Regress every unit's outcomes on each predictor, with an intercept and the standardised covariates.

    Args:
        covariates: genes x k, without an intercept column.
        predictors: genes x biosamples.
        outcomes: the genes x units outcomes.
        standard_errors: as for `Design`.
        regions: one label per gene, the clusters for `region` standard errors.
        max_bytes: size of the largest array a batch of units holds.

    Yields:
        The results, a batch of units at a time, in unit order.
    """
    n_genes = predictors.shape[0]
    clusters = None if regions is None else Clusters.from_labels(regions)
    design = Design(
        np.column_stack([np.ones(n_genes), standardise(covariates)]),
        predictors,
        standard_errors,
        clusters,
        max_bytes=max_bytes,
    )
    batch = max(1, max_bytes // (8 * n_genes))
    order = np.argsort(outcomes.units, kind='stable')
    genes, units, values = outcomes.genes[order], outcomes.units[order], outcomes.values[order]
    bounds = np.searchsorted(units, np.arange(0, outcomes.n_units + batch, batch))
    for i, start in enumerate(range(0, outcomes.n_units, batch)):
        stop = min(start + batch, outcomes.n_units)
        rows = slice(bounds[i], bounds[i + 1])
        y = np.zeros((n_genes, stop - start))
        y[genes[rows], units[rows] - start] = values[rows]
        fit = design.fit(y)
        with np.errstate(divide='ignore', invalid='ignore'):
            z = fit.beta / fit.standard_error
        z = np.where(np.isfinite(z), z, np.nan)
        yield Enrichment(start, fit.beta, fit.standard_error, upper_tail_p(z))
