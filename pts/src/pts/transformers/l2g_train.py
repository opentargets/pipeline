"""Train the locus-to-gene classifier from the pinned train/test split.

Replaces gentropy's `LocusToGeneTrainTestSplitStep` and the train mode of `LocusToGeneStep`. Three
things gentropy does are deliberately absent:

* **Cross-validation.** Its folds fed nothing back -- the estimator was cloned, fitted, logged and
  discarded, and the W&B "sweep" grid was built from the model's own single hyperparameter values,
  so it tuned nothing either. It also re-split the training set five times.
* **The Hugging Face Hub upload**, which called `generate_train_test_split` a second time purely to
  attach train/test parquets to the model repo.
* **Weights & Biases.** The metrics it carried are written to `metrics.json` in the release instead,
  versioned with the run that produced them.
"""

import json
from pathlib import Path
from typing import Any

import polars as pl
from loguru import logger
from otter.config.model import Config

from pts.transformers.l2g import explain, gold_standard, split
from pts.transformers.l2g import model as l2g_model
from pts.transformers.l2g.features import FEATURES
from pts.transformers.utils.dataset import scan_dataset, write_dataset


def _write_json(document: dict[str, Any], path: str) -> None:
    """Write a JSON document, creating the parent directory.

    Args:
        document: the object to serialise.
        path: destination file.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(document, indent=2))


def l2g_train(
    source: dict[str, str],
    destination: dict[str, str],
    settings: dict[str, Any],
    config: Config,
) -> None:
    """Build the pinned splits, fit the classifier and write the model and its metrics.

    Args:
        source: keys `feature_matrix`, `credible_set`, `gold_standard`, `predefined_test`.
        destination: keys `model`, `background`, `metrics`, `train_split`, `test_split`,
            `split_stats`.
        settings: keys `features_list`, `hyperparameters`, `train_on_full_dataset`,
            `shap_background_size`, `shap_background_seed`.
        config: otter config; unused, accepted for interface compatibility.
    """
    features = list(settings.get('features_list') or FEATURES)
    hyperparameters = dict(settings['hyperparameters'])
    background_size = int(settings['shap_background_size'])
    background_seed = int(settings['shap_background_seed'])

    logger.info('annotating the gold standard with feature-matrix rows')
    annotated = gold_standard.annotate(
        scan_dataset(source['feature_matrix']),
        scan_dataset(source['credible_set']),
        gold_standard.parse_gold_standard(scan_dataset(source['gold_standard'], format='ndjson')),
        features,
    ).collect()

    predefined = scan_dataset(source['predefined_test']).collect()
    train, test = split.derive_splits(annotated.lazy(), predefined.lazy())
    logger.info(f'split: {train.height} train rows, {test.height} test rows')

    stats = split.split_stats(annotated.height, predefined.height, train, test)
    _write_json(stats, destination['split_stats'])
    write_dataset(train, destination['train_split'])
    write_dataset(test, destination['test_split'])

    x_train = l2g_model.to_matrix(train, features)
    y_train = l2g_model.to_labels(train)
    x_test = l2g_model.to_matrix(test, features)
    y_test = l2g_model.to_labels(test)

    logger.info('fitting the classifier')
    fitted = l2g_model.fit(x_train, y_train, hyperparameters)
    held_out = l2g_model.evaluate(fitted, x_test, y_test)
    logger.info(f'held-out metrics: {held_out}')

    labelled = pl.concat([train, test], how='vertical')
    if settings.get('train_on_full_dataset'):
        # Evaluation above is complete and is not affected by this refit; the saved model simply
        # benefits from the held-out rows too, which is what produced the 26.09-2 model.
        logger.info('refitting on train + held-out for the saved model')
        fitted = l2g_model.fit(
            l2g_model.to_matrix(labelled, features), l2g_model.to_labels(labelled), hyperparameters
        )

    background = explain.build_background(labelled, features, background_size, background_seed)
    Path(destination['background']).parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(background, schema=features).write_parquet(destination['background'])

    l2g_model.save_model(fitted, destination['model'])
    _write_json(
        {
            'heldOut': held_out,
            'featureMissingness': l2g_model.missingness(labelled, features),
            'hyperparameters': hyperparameters,
            'features': features,
            'trainOnFullDataset': bool(settings.get('train_on_full_dataset')),
            'shapBackgroundSize': background_size,
            'shapBackgroundSeed': background_seed,
            'split': stats,
        },
        destination['metrics'],
    )
    logger.info('training complete')
