"""Interventional TreeSHAP in probability space, spread across processes.

Two things about this are deliberate and easy to get wrong.

First, the masker is constructed explicitly with `max_samples`. Passing a bare array as `data=`
lets `shap` wrap it in an `Independent` masker whose default `max_samples` is 100, which silently
discards most of a larger background -- that is exactly what gentropy does, so its 1000-row sample
has always been a 100-row background. Cost is linear in the size actually used: 181-317
rows/s/process at 100, 32 rows/s/process at 1000. The spread at 100 is machine load, not the
approach; 181 is the figure production ships and the one to size against.

Second, one background serves the whole run. gentropy draws an unseeded sample inside each of its
1000 Batch tasks, so `shapBaseValue` varies across output partitions: measured across all 200
partitions of `do/platform-2609-1`, 200 distinct base values from 0.028042 to 0.137713, a 4.91x
spread within a single released dataset. One seeded background gives one base value.
"""

import math
import multiprocessing
import os
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np
import polars as pl
import shap
import shap.maskers

_EXPLAINER: Any = None
"""Per-worker explainer, built once in the pool initialiser."""

DEFAULT_CHUNKS_PER_WORKER = 8
MIN_CHUNK_ROWS = 1_000
MAX_CHUNK_ROWS = 50_000


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


def _chunk_size(n_rows: int, workers: int, *, per_worker: int = DEFAULT_CHUNKS_PER_WORKER) -> int:
    """Rows per unit of work, sized so every worker gets several chunks.

    A chunk count below the worker count caps the speedup at the chunk count no matter how many
    cores the machine has, because a chunk cannot be split. Aiming for several chunks each also
    absorbs the uneven tail: 65 chunks over 32 workers has a makespan of 3 chunk-times against
    an ideal 2.03, wasting 48% of the wall clock, while 258 chunks over 32 wastes 12%.

    The floor keeps per-chunk process overhead small relative to the work -- at roughly 180
    rows/s a 1,000-row chunk is several seconds of computation. The cap bounds peak memory.
    """
    if n_rows <= 0:
        return MAX_CHUNK_ROWS
    return max(MIN_CHUNK_ROWS, min(MAX_CHUNK_ROWS, math.ceil(n_rows / (workers * per_worker))))


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
    chunk_size: int | None = None,
) -> tuple[float, np.ndarray]:
    """Compute SHAP values for every row, in parallel.

    The pool is forced onto the `spawn` start method; see the `ProcessPoolExecutor` below. A
    consequence for callers: under `spawn` each worker re-imports the caller's `__main__`, so an
    ad-hoc script that calls this at module scope needs an `if __name__ == "__main__":` guard or
    it will recurse. Production is safe -- otter's console script carries its own guard -- but an
    unguarded script does not, and diagnosing it costs more time than the guard.

    Args:
        model: the fitted classifier.
        matrix: rows to explain, in fitted feature order.
        background: the background array from `build_background`.
        max_samples: background rows the masker may use.
        workers: process count; defaults to `os.cpu_count()`.
        chunk_size: rows per unit of work; defaults to `_chunk_size` of the matrix and worker count.

    Returns:
        `(base_value, shap_values)` where `shap_values` has the shape of `matrix`.
    """
    # The parent's own explainer is built solely to read `expected_value`, and is retained
    # deliberately: under `spawn` it is not a fork hazard, and obtaining the base value any other
    # way would mean restructuring this function to save a few milliseconds.
    _initialise(model, background, max_samples)
    base_value = float(_EXPLAINER.expected_value)

    if matrix.shape[0] == 0:
        return base_value, np.empty_like(matrix)

    resolved_workers = workers or os.cpu_count() or 1
    if chunk_size is None:
        chunk_size = _chunk_size(matrix.shape[0], resolved_workers)

    chunks = [matrix[i : i + chunk_size] for i in range(0, matrix.shape[0], chunk_size)]

    if resolved_workers == 1:
        return base_value, np.vstack([_explain_chunk(chunk) for chunk in chunks])

    # `spawn` is pinned rather than left to the platform default, which is spawn on macOS but
    # FORK on Linux -- and production is `python:3.11-slim`. By this point the parent has already
    # run `predict_proba` over the whole matrix and built a `TreeExplainer`, both heavy OpenMP
    # regions; forking a live OpenMP runtime and then entering a parallel region in the child is
    # the classic hang, and no PTS step sets `execution_timeout`, so a hung pool runs until a
    # human notices. Every test and every verification of this path has run on macOS, i.e. under
    # spawn, so this also makes production execute the only configuration that has been exercised.
    # The cost is one module re-import per worker: the model and background already travel through
    # `initargs`, so nothing else has to be re-derived.
    with ProcessPoolExecutor(
        max_workers=resolved_workers,
        mp_context=multiprocessing.get_context('spawn'),
        initializer=_initialise,
        initargs=(model, background, max_samples),
    ) as pool:
        return base_value, np.vstack(list(pool.map(_explain_chunk, chunks)))
