"""Fit, evaluate and persist the L2G classifier.

`DEFAULT_HYPERPARAMETERS` are the values gentropy was pinned to in #123, in full, so a change
of gentropy or XGBoost defaults cannot change the model.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
import polars as pl
import skops.io as sio
from otter.storage.synchronous.handle import StorageHandle
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from pts.transformers.l2g.gold_standard import LABEL_COLUMN
from pts.transformers.l2g.split import UNIT_COLUMN

DEFAULT_HYPERPARAMETERS: dict[str, Any] = {
    'objective': 'binary:logistic',
    'eval_metric': 'aucpr',
    'random_state': 777,
    'n_estimators': 200,
    'max_depth': 4,
    'min_child_weight': 10,
    'eta': 0.025,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 1,
    'reg_lambda': 1.0,
    'scale_pos_weight': 0.8,
    'gamma': 0,
    'max_delta_step': 1,
}
"""Pinned in #123; see the module docstring."""


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


def fit(
    x: np.ndarray,
    y: np.ndarray,
    hyperparameters: dict[str, Any],
    sample_weight: np.ndarray | None = None,
) -> XGBClassifier:
    """Fit the classifier.

    Args:
        x: feature matrix.
        y: integer labels.
        hyperparameters: passed straight to `XGBClassifier`.
        sample_weight: one weight per row, or None to fit unweighted.

    Returns:
        The fitted classifier.
    """
    model = XGBClassifier(**hyperparameters)
    model.fit(X=x, y=y, sample_weight=sample_weight)
    return model


DEFAULT_UNIT_WEIGHT_K = 5.0
"""Credible sets per association unit counted at full weight; see `association_unit_weights`."""


def association_unit_weights(frame: pl.DataFrame, k: float | None = DEFAULT_UNIT_WEIGHT_K) -> np.ndarray:
    """Down-weight credible sets that repeat the same association.

    The same association is fine-mapped in many studies, so the labelled set repeats near-identical
    credible sets. With n credible sets in a unit, each gets min(1, sqrt(k / n)): units of up to k
    credible sets keep full weight and larger ones are down-weighted gently, so a unit of 50 counts
    about 3 times one credible set rather than 50 times. `k=None` gives 1/n, so each unit counts
    once, as in gentropy `LocusToGeneTrainer.association_unit_weights` (opentargets/gentropy#1314).
    On the 26.09 labels 1/n raised scores of far genes with no functional evidence and moved feature
    importance to gene density and E2G, while min(1, sqrt(5 / n)) kept the full feature-matrix
    score distribution and feature importance close to the unweighted model.

    All candidate-gene rows of a credible set share its weight. The weights are rescaled to a mean of
    1 over rows, so `min_child_weight` keeps its meaning. A credible set without a unit is its own
    unit.

    Args:
        frame: rows with `credibleSetId` and `associationUnitId`.
        k: credible sets per unit counted at full weight, or None for 1/n.

    Returns:
        One `float64` weight per row, in row order.
    """
    unit = pl.col(UNIT_COLUMN).fill_null(pl.col('credibleSetId'))
    n = frame.select(pl.col('credibleSetId').n_unique().over(unit).cast(pl.Float64).alias('n'))['n'].to_numpy()
    weights = 1.0 / n if k is None else np.minimum(1.0, np.sqrt(k / n))
    return weights / weights.mean()


def top_one_accuracy(
    credible_set_ids: Sequence[str], y: np.ndarray, scores: np.ndarray, sample_weight: np.ndarray | None = None
) -> float:
    """Share of positives that score highest among the genes of their credible set.

    Args:
        credible_set_ids: credible set of each row.
        y: true integer labels.
        scores: predicted probabilities of the positive class.
        sample_weight: one weight per row, or None for equal weights.

    Returns:
        The weighted share of positive rows ranked first in their credible set.
    """
    frame = pl.DataFrame({'cs': list(credible_set_ids), 'y': y, 's': scores})
    first = frame.select(pl.col('s').rank('ordinal', descending=True).over('cs') == 1)['s'].to_numpy()
    positive = y == 1
    weights = None if sample_weight is None else sample_weight[positive]
    return float(np.average(first[positive], weights=weights))


def evaluate(
    model: XGBClassifier,
    x: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray | None = None,
    credible_set_ids: Sequence[str] | None = None,
) -> dict[str, float]:
    """Evaluate a fitted model.

    `averagePrecision` is computed from the predicted scores, as gentropy computes it since
    opentargets/gentropy#1283. Before that change both gentropy and this module computed it from the
    hard 0/1 predictions, which gives one interior point on the precision-recall curve and an area
    far below the model's real value (0.485 vs 0.709 on the 2612-2 test split). That old value is
    still reported as `averagePrecisionFromLabels`, so it can be compared with metrics recorded
    before the change (W&B runs, 26.09 and earlier `metrics.json`).

    Args:
        model: a fitted classifier.
        x: feature matrix.
        y: true integer labels.
        sample_weight: one weight per row applied to every metric, or None for equal weights.
        credible_set_ids: credible set of each row; when given, `topOneAccuracy` is added.

    Returns:
        Metric name to value.
    """
    predicted = model.predict(x)
    scores = model.predict_proba(x)[:, 1]
    w = sample_weight
    metrics = {
        'areaUnderROC': float(roc_auc_score(y, scores, average='weighted', sample_weight=w)),
        'accuracy': float(accuracy_score(y, predicted, sample_weight=w)),
        'weightedPrecision': float(precision_score(y, predicted, average='weighted', zero_division=0, sample_weight=w)),
        'averagePrecision': float(average_precision_score(y, scores, sample_weight=w)),
        'averagePrecisionFromLabels': float(average_precision_score(y, predicted, sample_weight=w)),
        'weightedRecall': float(recall_score(y, predicted, average='weighted', zero_division=0, sample_weight=w)),
        'f1': float(f1_score(y, predicted, average='weighted', zero_division=0, sample_weight=w)),
    }
    if credible_set_ids is not None:
        metrics['topOneAccuracy'] = top_one_accuracy(credible_set_ids, y, scores, w)
    return metrics


def save_model(model: Any, path: str) -> None:
    """Persist a fitted model in the skops format, at the path the release expects.

    Serialises to bytes and writes through otter's storage abstraction rather than handing
    `skops` a path. `sio.dump` is `pathlib`-backed, and in production the destination is a
    `gs://…` URI that POSIX collapses to `gs:/`, so the model would land on the container's local
    disk while the step reported success -- and the prediction task, running in the same working
    directory, would read that local copy back and hide it.

    Args:
        model: the fitted classifier.
        path: destination, local or `gs://`, which must end in `.skops`.

    Raises:
        ValueError: if `path` does not end in `.skops`.
    """
    if not path.endswith('.skops'):
        msg = f'model path must end with .skops, got {path!r}'
        raise ValueError(msg)
    StorageHandle(path).write(sio.dumps(model))


def load_model(path: str) -> XGBClassifier:
    """Load a model persisted by `save_model`.

    The symmetric read: the bytes come through `StorageHandle`, so a model stored in the release
    bucket loads from its `gs://` URI instead of from whatever `gs:/…` tree happens to sit under
    the current working directory.

    `skops` refuses unknown types unless they are named as trusted, so the payload's own reported
    types are passed through. That is safe here because the file is written by this pipeline into
    the release it is read back from.

    Args:
        path: a `.skops` file, local or `gs://`.

    Returns:
        The classifier.
    """
    blob, _ = StorageHandle(path).read()
    return sio.loads(blob, trusted=sio.get_untrusted_types(data=blob))


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
