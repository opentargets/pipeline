"""Tests for SHAP background construction and the explainer pool."""

import numpy as np
import polars as pl
import pytest

from pts.transformers.l2g.explain import build_background, explain
from pts.transformers.l2g.model import DEFAULT_HYPERPARAMETERS, fit


@pytest.fixture
def model_and_data():
    rng = np.random.default_rng(0)
    x = np.vstack([rng.normal(0.0, 0.1, (60, 2)), rng.normal(3.0, 0.1, (60, 2))]).astype(np.float32)
    y = np.concatenate([np.zeros(60), np.ones(60)]).astype(np.int32)
    model = fit(x, y, {**DEFAULT_HYPERPARAMETERS, 'n_estimators': 5, 'max_depth': 2})
    return model, x


def test_build_background_returns_the_requested_size() -> None:
    frame = pl.DataFrame({'a': list(range(100)), 'b': list(range(100))}).cast(pl.Float32)
    assert build_background(frame, ['a', 'b'], 10, 42).shape == (10, 2)


def test_build_background_is_reproducible_for_a_fixed_seed() -> None:
    frame = pl.DataFrame({'a': list(range(100)), 'b': list(range(100))}).cast(pl.Float32)
    first = build_background(frame, ['a', 'b'], 10, 42)
    second = build_background(frame, ['a', 'b'], 10, 42)
    np.testing.assert_array_equal(first, second)


def test_build_background_differs_for_a_different_seed() -> None:
    frame = pl.DataFrame({'a': list(range(100)), 'b': list(range(100))}).cast(pl.Float32)
    assert not np.array_equal(
        build_background(frame, ['a', 'b'], 10, 1), build_background(frame, ['a', 'b'], 10, 2)
    )


def test_build_background_uses_every_row_when_the_frame_is_smaller_than_the_size() -> None:
    frame = pl.DataFrame({'a': [1.0, 2.0], 'b': [3.0, 4.0]}).cast(pl.Float32)
    assert build_background(frame, ['a', 'b'], 10, 42).shape == (2, 2)


def test_build_background_uses_the_requested_feature_order() -> None:
    frame = pl.DataFrame({'a': [1.0], 'b': [2.0]}).cast(pl.Float32)
    assert build_background(frame, ['b', 'a'], 1, 42).tolist() == [[2.0, 1.0]]


def test_explain_returns_one_shap_value_per_cell(model_and_data) -> None:
    model, x = model_and_data
    _, values = explain(model, x[:20], x[:10], max_samples=10, workers=1)
    assert values.shape == (20, 2)


def test_explain_returns_a_single_base_value(model_and_data) -> None:
    model, x = model_and_data
    base, _ = explain(model, x[:20], x[:10], max_samples=10, workers=1)
    assert isinstance(base, float)


def test_explain_chunking_does_not_change_the_result(model_and_data) -> None:
    model, x = model_and_data
    _, whole = explain(model, x[:20], x[:10], max_samples=10, workers=1, chunk_size=100)
    _, chunked = explain(model, x[:20], x[:10], max_samples=10, workers=1, chunk_size=3)
    np.testing.assert_allclose(whole, chunked, rtol=1e-6, atol=1e-9)


def test_explain_across_workers_matches_a_single_worker(model_and_data) -> None:
    model, x = model_and_data
    _, single = explain(model, x[:20], x[:10], max_samples=10, workers=1, chunk_size=5)
    _, parallel = explain(model, x[:20], x[:10], max_samples=10, workers=2, chunk_size=5)
    np.testing.assert_allclose(single, parallel, rtol=1e-6, atol=1e-9)


def test_explain_handles_an_empty_matrix(model_and_data) -> None:
    model, x = model_and_data
    _, values = explain(model, x[:0], x[:10], max_samples=10, workers=1)
    assert values.shape == (0, 2)
