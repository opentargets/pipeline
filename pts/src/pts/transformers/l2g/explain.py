"""Interventional TreeSHAP in probability space, spread across processes.

Two things about this are deliberate and easy to get wrong.

First, the masker is constructed explicitly with `max_samples`. Passing a bare array as `data=`
lets `shap` wrap it in an `Independent` masker whose default `max_samples` is 100, which silently
discards most of a larger background -- that is exactly what gentropy does, so its 1000-row sample
has always been a 100-row background. Cost is linear in the size actually used: 317 rows/s/process
at 100, 32 rows/s/process at 1000.

Second, one background serves the whole run. gentropy draws an unseeded sample inside each of its
1000 Batch tasks, so `shapBaseValue` varies across output partitions -- 0.0381 to 0.0668 in
26.09-1. One seeded background gives one base value.
"""

from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np
import polars as pl
import shap
import shap.maskers

_EXPLAINER: Any = None
"""Per-worker explainer, built once in the pool initialiser."""


def build_background(
    frame: pl.DataFrame,
    features: Sequence[str],
    size: int,
    seed: int,
) -> np.ndarray:
    """Draw the SHAP background from the labelled data.

    Args:
        frame: the rows to sample from, normally train and test concatenated.
        features: feature names, in fitted order.
        size: how many rows to draw; the whole frame is used if it holds fewer.
        seed: seed for the draw, recorded in the run's metrics so it can be reproduced.

    Returns:
        A `float32` array of shape `(min(size, rows), len(features))`.
    """
    matrix = frame.select(features).to_numpy().astype(np.float32)
    if matrix.shape[0] <= size:
        return matrix
    rng = np.random.default_rng(seed)
    return matrix[rng.choice(matrix.shape[0], size, replace=False)]


def _initialise(model: Any, background: np.ndarray, max_samples: int) -> None:
    """Build this worker's explainer.

    Args:
        model: the fitted classifier.
        background: the shared background array.
        max_samples: how many background rows the masker may use.
    """
    global _EXPLAINER  # noqa: PLW0603 -- one explainer per worker process, built once
    masker = shap.maskers.Independent(background, max_samples=max_samples)
    _EXPLAINER = shap.TreeExplainer(
        model,
        data=masker,  # ty: ignore[invalid-argument-type] -- shap's stub predates masker-typed `data`
        feature_perturbation='interventional',
        model_output='probability',
    )


def _explain_chunk(chunk: np.ndarray) -> np.ndarray:
    """Compute SHAP values for one chunk of rows.

    Args:
        chunk: rows to explain.

    Returns:
        SHAP values, same shape as `chunk`.
    """
    return np.asarray(_EXPLAINER.shap_values(chunk, check_additivity=False))


def explain(
    model: Any,
    matrix: np.ndarray,
    background: np.ndarray,
    *,
    max_samples: int,
    workers: int | None = None,
    chunk_size: int = 50_000,
) -> tuple[float, np.ndarray]:
    """Compute SHAP values for every row, in parallel.

    Args:
        model: the fitted classifier.
        matrix: rows to explain, in fitted feature order.
        background: the background array from `build_background`.
        max_samples: background rows the masker may use.
        workers: process count; defaults to the pool's own default.
        chunk_size: rows per unit of work.

    Returns:
        `(base_value, shap_values)` where `shap_values` has the shape of `matrix`.
    """
    _initialise(model, background, max_samples)
    base_value = float(_EXPLAINER.expected_value)

    if matrix.shape[0] == 0:
        return base_value, np.empty_like(matrix)

    chunks = [matrix[i : i + chunk_size] for i in range(0, matrix.shape[0], chunk_size)]

    if workers == 1:
        return base_value, np.vstack([_explain_chunk(chunk) for chunk in chunks])

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_initialise,
        initargs=(model, background, max_samples),
    ) as pool:
        return base_value, np.vstack(list(pool.map(_explain_chunk, chunks)))
