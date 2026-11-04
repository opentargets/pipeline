"""Block jackknife for least squares, as LDSC estimates its standard errors.

A port of `ldscore/jackknife.py` and `ldscore/irwls.py` from github.com/bulik/ldsc (Bulik-Sullivan
et al. 2015, doi:10.1038/ng.3406), kept to the pieces the single-annotation regressions use.

SNPs are split into contiguous blocks. Each block's contribution to X'X and X'y is summed once,
so leaving a block out costs one small solve rather than a refit over all SNPs. The delete values
(the estimate with one block left out) turn into pseudovalues, whose spread gives the standard
error; they are also what a ratio of two estimates, like rg, is jackknifed from.

Arrays are 2-D as in LDSC: rows are SNPs (or blocks), columns parameters.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

_IRWLS_ITERATIONS = 2
"""Weight updates before the final weighted fit, as LDSC does it."""


@dataclass(frozen=True)
class Jackknife:
    """A least squares fit and its block jackknife."""

    est: np.ndarray
    """Whole-data estimate, shape (1, p)."""
    delete_values: np.ndarray
    """Estimate with each block left out, shape (n_blocks, p)."""
    jackknife_est: np.ndarray
    """Mean of the pseudovalues, shape (1, p)."""
    jackknife_se: np.ndarray
    """Standard error of each parameter, shape (1, p)."""
    jackknife_cov: np.ndarray
    """Covariance of the parameters, shape (p, p)."""
    separators: np.ndarray
    """Block boundaries, `n_blocks + 1` row indices from 0 to n."""


def separators(n: int, n_blocks: int) -> np.ndarray:
    """Evenly spaced block boundaries over `n` rows."""
    return np.floor(np.linspace(0, n, n_blocks + 1)).astype(int)


def unmask_separators(masked: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Map separators found on the rows where `mask` is true back to the full row index."""
    rows = np.flatnonzero(mask)
    return np.concatenate([[0], rows[masked[1:-1]], [len(mask)]])


def pseudovalues(delete_values: np.ndarray, est: np.ndarray) -> np.ndarray:
    """Jackknife pseudovalues from delete values and the whole-data estimate."""
    n_blocks = delete_values.shape[0]
    return n_blocks * est - (n_blocks - 1) * delete_values


def summarise(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The jackknife estimate, standard errors and covariance from pseudovalues (n_blocks, p)."""
    n_blocks = values.shape[0]
    cov = np.atleast_2d(np.cov(values.T, ddof=1) / n_blocks)
    est = np.mean(values, axis=0).reshape((1, -1))
    return est, np.sqrt(np.diag(cov)).reshape((1, -1)), cov


def least_squares(x: np.ndarray, y: np.ndarray, boundaries: np.ndarray) -> Jackknife:
    """Fit `y ~ x` and jackknife the fit over the blocks `boundaries` delimit.

    Args:
        x: design, shape (n, p).
        y: response, shape (n, 1).
        boundaries: block separators, from `separators` or `unmask_separators`.

    Returns:
        The fit.

    Raises:
        ValueError: with more blocks than rows.
    """
    n, p = x.shape
    n_blocks = len(boundaries) - 1
    if n_blocks > n:
        raise ValueError(f'{n_blocks} jackknife blocks but only {n} rows')
    xty = np.zeros((n_blocks, p))
    xtx = np.zeros((n_blocks, p, p))
    for i in range(n_blocks):
        rows = slice(boundaries[i], boundaries[i + 1])
        xty[i] = (x[rows].T @ y[rows]).reshape(p)
        xtx[i] = x[rows].T @ x[rows]
    xty_total = xty.sum(axis=0)
    xtx_total = xtx.sum(axis=0)
    est = np.linalg.solve(xtx_total, xty_total).reshape((1, p))
    delete_values = np.zeros((n_blocks, p))
    for i in range(n_blocks):
        delete_values[i] = np.linalg.solve(xtx_total - xtx[i], xty_total - xty[i])
    jackknife_est, se, cov = summarise(pseudovalues(delete_values, est))
    return Jackknife(est, delete_values, jackknife_est, se, cov, np.asarray(boundaries))


def ratio(est: float, numerator: np.ndarray, denominator: np.ndarray) -> tuple[float, float]:
    """Jackknife a ratio of two estimates from their delete values, shape (n_blocks,) each.

    Args:
        est: the whole-data ratio.
        numerator: delete values of the numerator.
        denominator: delete values of the denominator.

    Returns:
        The jackknife estimate of the ratio and its standard error. Both NaN when a denominator
        delete value is not a positive finite number.
    """
    n_blocks = len(numerator)
    with np.errstate(divide='ignore', invalid='ignore'):
        values = n_blocks * est - (n_blocks - 1) * numerator / denominator
    if not np.all(np.isfinite(values)):
        return math.nan, math.nan
    jackknife_est, se, _ = summarise(values.reshape((n_blocks, 1)))
    return float(jackknife_est[0, 0]), float(se[0, 0])


def weighted(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Scale the rows of `x` by `w` normalised to sum to one, as LDSC weights its designs."""
    if np.any(w <= 0):
        raise ValueError('weights must be positive')
    return x * (w / w.sum())


def weighted_least_squares(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Coefficients of `y ~ x` with row weights `w` (already square-rooted), shape (p, 1)."""
    return np.linalg.lstsq(weighted(x, w), weighted(y, w), rcond=None)[0]


def reweighted_least_squares(
    x: np.ndarray,
    y: np.ndarray,
    update: Callable[[np.ndarray], np.ndarray],
    w: np.ndarray,
    boundaries: np.ndarray,
) -> Jackknife:
    """Iteratively reweighted least squares, then a block jackknife of the final weighted fit.

    Args:
        x: design, shape (n, p).
        y: response, shape (n, 1).
        update: new inverse-variance weights from the coefficients of a weighted fit.
        w: initial inverse-variance weights, shape (n, 1).
        boundaries: jackknife block separators.

    Returns:
        The jackknife of the final weighted fit.
    """
    w = np.sqrt(w)
    for _ in range(_IRWLS_ITERATIONS):
        w = np.sqrt(update(weighted_least_squares(x, y, w)))
    return least_squares(weighted(x, w), weighted(y, w), boundaries)
