"""Fit, evaluate and persist the L2G classifier.

`DEFAULT_HYPERPARAMETERS` is copied from the model that shipped with 26.09-2, read off
`classifier.skops` itself. Three of its entries -- `n_estimators`, `gamma` and `max_delta_step` --
appear in neither gentropy nor this repository, so reading them from gentropy's config would give
a 100-tree model where the released one has 300. They live here so that never happens again.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# `xgboost` must be imported (and so initialize its bundled OpenMP runtime) before `skops.io` --
# `skops.io` pulls in sklearn's compiled tree/SGD extensions, and if those load their own OpenMP
# copy first, the process later segfaults inside `XGBClassifier.fit`/`predict` on macOS. Keep this
# import above `skops.io` even though it breaks alphabetical order.
from xgboost import XGBClassifier  # isort: skip

import skops.io as sio

from pts.transformers.l2g.gold_standard import LABEL_COLUMN

DEFAULT_HYPERPARAMETERS: dict[str, Any] = {
    'objective': 'binary:logistic',
    'eval_metric': 'aucpr',
    'random_state': 777,
    'n_estimators': 300,
    'max_depth': 5,
    'min_child_weight': 10,
    'eta': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 1,
    'reg_lambda': 1.0,
    'scale_pos_weight': 0.8,
    'gamma': 0,
    'max_delta_step': 1,
}
"""Read from the 26.09-2 `classifier.skops`; see the module docstring."""


def to_matrix(frame: pl.DataFrame, features: Sequence[str]) -> np.ndarray:
    """Extract the feature matrix in fitted order.

    Args:
        frame: a partition carrying at least the feature columns.
        features: feature names, in the order the model expects.

    Returns:
        A `float32` array of shape `(rows, len(features))`.
    """
    return frame.select(features).to_numpy().astype(np.float32)


def to_labels(frame: pl.DataFrame, column: str = LABEL_COLUMN) -> np.ndarray:
    """Extract the integer label vector.

    Args:
        frame: a partition carrying the label column.
        column: name of the label column.

    Returns:
        An `int32` array.
    """
    return frame[column].to_numpy().astype(np.int32)


def fit(x: np.ndarray, y: np.ndarray, hyperparameters: dict[str, Any]) -> XGBClassifier:
    """Fit the classifier.

    Args:
        x: feature matrix.
        y: integer labels.
        hyperparameters: passed straight to `XGBClassifier`.

    Returns:
        The fitted classifier.
    """
    model = XGBClassifier(**hyperparameters)
    model.fit(X=x, y=y)
    return model


def evaluate(model: XGBClassifier, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Evaluate a fitted model, reporting the six metrics gentropy reports.

    Args:
        model: a fitted classifier.
        x: feature matrix.
        y: true integer labels.

    Returns:
        Metric name to value.
    """
    predicted = model.predict(x)
    probabilities = model.predict_proba(x)
    return {
        'areaUnderROC': float(roc_auc_score(y, probabilities[:, 1], average='weighted')),
        'accuracy': float(accuracy_score(y, predicted)),
        'weightedPrecision': float(precision_score(y, predicted, average='weighted', zero_division=0)),
        'averagePrecision': float(average_precision_score(y, probabilities[:, 1], average='weighted')),
        'weightedRecall': float(recall_score(y, predicted, average='weighted', zero_division=0)),
        'f1': float(f1_score(y, predicted, average='weighted', zero_division=0)),
    }


def save_model(model: Any, path: str) -> None:
    """Persist a fitted model in the skops format, at the path the release expects.

    Args:
        model: the fitted classifier.
        path: destination, which must end in `.skops`.

    Raises:
        ValueError: if `path` does not end in `.skops`.
    """
    if not path.endswith('.skops'):
        msg = f'model path must end with .skops, got {path!r}'
        raise ValueError(msg)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sio.dump(model, path)


def load_model(path: str) -> XGBClassifier:
    """Load a model persisted by `save_model`.

    `skops` refuses unknown types unless they are named as trusted, so the file's own reported
    types are passed through. That is safe here because the file is written by this pipeline into
    the release it is read back from.

    Args:
        path: a `.skops` file.

    Returns:
        The classifier.
    """
    return sio.load(path, trusted=sio.get_untrusted_types(file=path))


def missingness(frame: pl.DataFrame, features: Sequence[str]) -> dict[str, float]:
    """Fraction of rows where each feature is null or zero.

    Mirrors `L2GFeatureMatrix.calculate_feature_missingness_rate`, which counts zero as missing
    because the imputation has already turned nulls into zeros by the time it runs.

    Args:
        frame: the partition to measure.
        features: feature names.

    Returns:
        Feature name to fraction in [0, 1].
    """
    total = frame.height
    if total == 0:
        return dict.fromkeys(features, 0.0)
    return {
        name: frame.filter(pl.col(name).is_null() | (pl.col(name) == 0)).height / total
        for name in features
    }
