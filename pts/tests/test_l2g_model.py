"""Tests for L2G model fitting, evaluation and persistence."""

import numpy as np
import polars as pl
import pytest

from pts.transformers.l2g.model import (
    DEFAULT_HYPERPARAMETERS,
    evaluate,
    fit,
    load_model,
    missingness,
    save_model,
    to_labels,
    to_matrix,
)


@pytest.fixture
def separable() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    x = np.vstack([rng.normal(0.0, 0.1, (60, 2)), rng.normal(3.0, 0.1, (60, 2))]).astype(np.float32)
    y = np.concatenate([np.zeros(60), np.ones(60)]).astype(np.int32)
    return x, y


@pytest.fixture
def overlapping() -> tuple[np.ndarray, np.ndarray]:
    """Features carry no real signal, so the model does not perfectly separate the classes --
    unlike `separable`, where every metric collapses to 1.0 regardless of formula.
    """
    rng = np.random.default_rng(1)
    x = rng.normal(0.0, 1.0, (200, 2)).astype(np.float32)
    y = (rng.random(200) < 0.5).astype(np.int32)
    return x, y


def test_default_hyperparameters_match_the_released_model() -> None:
    assert DEFAULT_HYPERPARAMETERS['n_estimators'] == 300
    assert DEFAULT_HYPERPARAMETERS['random_state'] == 777
    assert DEFAULT_HYPERPARAMETERS['max_delta_step'] == 1
    assert DEFAULT_HYPERPARAMETERS['gamma'] == 0


def test_to_matrix_uses_the_requested_feature_order() -> None:
    frame = pl.DataFrame({'a': [1.0], 'b': [2.0], 'studyLocusId': ['sl1']})
    assert to_matrix(frame, ['b', 'a']).tolist() == [[2.0, 1.0]]


def test_to_matrix_returns_float32() -> None:
    frame = pl.DataFrame({'a': [1.0]})
    assert to_matrix(frame, ['a']).dtype == np.float32


def test_to_labels_returns_integers() -> None:
    frame = pl.DataFrame({'goldStandardSet': [0, 1]})
    assert to_labels(frame).tolist() == [0, 1]


def test_fit_produces_a_model_with_the_configured_number_of_trees(separable) -> None:
    x, y = separable
    model = fit(x, y, {**DEFAULT_HYPERPARAMETERS, 'n_estimators': 7})
    assert model.get_booster().num_boosted_rounds() == 7


def test_fit_is_deterministic_for_a_fixed_seed(separable) -> None:
    x, y = separable
    first = fit(x, y, {**DEFAULT_HYPERPARAMETERS, 'n_estimators': 7})
    second = fit(x, y, {**DEFAULT_HYPERPARAMETERS, 'n_estimators': 7})
    assert first.get_params()['random_state'] == 777
    np.testing.assert_array_equal(first.predict_proba(x), second.predict_proba(x))


def test_evaluate_reports_the_seven_metrics(separable) -> None:
    x, y = separable
    model = fit(x, y, {**DEFAULT_HYPERPARAMETERS, 'n_estimators': 7})
    metrics = evaluate(model, x, y)
    assert set(metrics) == {
        'areaUnderROC',
        'accuracy',
        'weightedPrecision',
        'averagePrecision',
        'averagePrecisionFromScores',
        'weightedRecall',
        'f1',
    }
    assert metrics['accuracy'] == pytest.approx(1.0)


def test_average_precision_and_average_precision_from_scores_differ(overlapping) -> None:
    x, y = overlapping
    model = fit(x, y, {**DEFAULT_HYPERPARAMETERS, 'n_estimators': 7})
    metrics = evaluate(model, x, y)
    assert metrics['averagePrecision'] != metrics['averagePrecisionFromScores']


def test_save_and_load_round_trip(separable, tmp_path) -> None:
    x, y = separable
    model = fit(x, y, {**DEFAULT_HYPERPARAMETERS, 'n_estimators': 7})
    path = tmp_path / 'model' / 'classifier.skops'
    save_model(model, str(path))
    assert path.exists()
    np.testing.assert_array_equal(load_model(str(path)).predict_proba(x), model.predict_proba(x))


def test_save_model_rejects_a_path_that_is_not_skops(tmp_path) -> None:
    with pytest.raises(ValueError, match=r'\.skops'):
        save_model(object(), str(tmp_path / 'classifier.json'))


def test_missingness_counts_null_and_zero_as_missing() -> None:
    frame = pl.DataFrame({'a': [0.0, 1.0, 2.0, 3.0], 'b': [1.0, 1.0, 1.0, 1.0]})
    assert missingness(frame, ['a', 'b']) == {'a': 0.25, 'b': 0.0}
