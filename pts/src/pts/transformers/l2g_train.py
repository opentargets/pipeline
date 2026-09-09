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
from typing import Any

import polars as pl
from loguru import logger
from otter.config.model import Config
from otter.storage.synchronous.handle import StorageHandle

from pts.transformers.l2g import explain, gold_standard, split
from pts.transformers.l2g import model as l2g_model
from pts.transformers.l2g.features import FEATURES
from pts.transformers.utils.dataset import scan_dataset, write_dataset


def _write_json(document: dict[str, Any], path: str) -> None:
    """Write a JSON document through otter's storage abstraction.

    Deliberately NOT `pathlib`. In production `release_uri` is set, so otter resolves every
    relative destination in `config.yaml` into a `gs://…` URI before this transformer sees it.
    POSIX collapses that scheme to `gs:/` and a `Path(...).write_text` lands the document on the
    container's local disk, silently, while the step still reports success. `StorageHandle`
    speaks both schemes, and its filesystem backend creates the parent directory itself.

    Args:
        document: the object to serialise.
        path: destination file, local or `gs://`.
    """
    StorageHandle(path).write_text(json.dumps(document, indent=2))


SORT_KEY: tuple[str, ...] = ('studyLocusId', 'geneId')
"""The natural key the splits are ordered on, which is only an order if it is unique."""


def require_unique_key(frame: pl.DataFrame, partition: str) -> None:
    """Refuse a split whose natural key repeats, because the sort would not order it.

    The sort on `SORT_KEY` is what makes training reproducible, and it is a TOTAL order only
    while the key is unique. Polars' `sort` is not tie-stable, and `maintain_order=True` would
    not help: it stabilises against the INPUT order, which is exactly the thing the joins in
    `derive_splits` leave unspecified. So a repeated pair puts the tied rows back in an arbitrary
    relative order and the positional draws below -- `subsample` and the SHAP background -- go
    back to varying between runs.

    That failure is invisible from the outside: `metrics.json` still records its seeds, the
    reproducibility test still passes on a key-unique fixture, and the only symptom is a
    different model every release with nothing to say why. A failed run is much the better
    outcome, so this raises.

    Args:
        frame: a train or test partition.
        partition: which one, for the message.

    Raises:
        ValueError: if any `SORT_KEY` pair occurs more than once.
    """
    keys = frame.select(SORT_KEY)
    duplicated = keys.filter(keys.is_duplicated()).unique(maintain_order=True)
    if duplicated.height:
        examples = duplicated.head(5).rows()
        msg = (
            f'the {partition} split repeats {duplicated.height} {SORT_KEY} pair(s), e.g. {examples}. '
            'Training sorts on that key to stay reproducible, and the sort is only an order while '
            'the key is unique -- tied rows would be left in the arbitrary order the joins in '
            'derive_splits produced, and the model and SHAP background would vary between runs '
            'again. Deduplicate the gold standard, or widen the key here and in the sort.'
        )
        raise ValueError(msg)


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
    # Polars joins leave their output order unspecified (`maintain_order=None`), and every path
    # into `derive_splits` ends in one, so the same inputs come back in a different row order from
    # run to run -- observed three orders in three calls within a single process. Everything below
    # this line is positional: XGBoost's `subsample` draws its rows by POSITION, and
    # `build_background` samples the background by position too. Sorting on the natural key, which
    # is unique here, is what makes the fitted model and the SHAP background reproducible from the
    # seeds `metrics.json` records; without it `random_state` and `shapBackgroundSeed` pin a draw
    # over rows that are not the same rows. `require_unique_key` then refuses the run if that
    # uniqueness ever stops holding, since the sort would silently stop being an order.
    train = train.sort(SORT_KEY)
    test = test.sort(SORT_KEY)
    require_unique_key(train, 'train')
    require_unique_key(test, 'test')
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

    # Sorted, because both partitions are: `vertical` preserves each frame's order, so the refit
    # below and the positional background draw further down both see a deterministic row order.
    labelled = pl.concat([train, test], how='vertical')
    if settings.get('train_on_full_dataset'):
        # Evaluation above is complete and is not affected by this refit; the saved model simply
        # benefits from the held-out rows too, which is what produced the 26.09-2 model.
        logger.info('refitting on train + held-out for the saved model')
        fitted = l2g_model.fit(
            l2g_model.to_matrix(labelled, features), l2g_model.to_labels(labelled), hyperparameters
        )

    background = explain.build_background(labelled, features, background_size, background_seed)
    # Deliberately NOT write_dataset: a single named release artifact, not a dataset of parts --
    # the same trade `release_metrics` makes for `metrics.parquet`. `mkdir=True` creates the
    # parent locally and is inert on a cloud URI, which polars writes natively; a
    # `Path(...).parent.mkdir` here would instead manufacture a stray local `gs:/…` tree.
    pl.DataFrame(background, schema=features).write_parquet(destination['background'], mkdir=True)

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
