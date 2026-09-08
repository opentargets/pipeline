"""Score every GWAS credible-set/gene pair and explain the survivors.

Replaces the predict mode of gentropy's `LocusToGeneStep`, which ran as 1000 Google Batch tasks.
Each of those started a Spark job on two vCPUs and re-read the ENTIRE feature matrix -- only the
credible-set side was partitioned -- to emit roughly 3,200 rows. The measured work is ~12 s of
scoring for all 61.2M rows plus ~2.8 process-hours of SHAP for the 3.2M that survive the
threshold, so it fits on one VM with room to spare.

Unlike gentropy this never drops the features and re-joins the matrix to get them back: they are
already on the frame that produced the score.
"""

import os
from collections.abc import Sequence
from typing import Any

import numpy as np
import polars as pl
from loguru import logger
from otter.config.model import Config

from pts.transformers.l2g import explain as l2g_explain
from pts.transformers.l2g import model as l2g_model
from pts.transformers.l2g.features import FEATURES, impute_and_cast
from pts.transformers.utils.dataset import scan_dataset, write_dataset

OUTPUT_COLUMNS = ('studyLocusId', 'geneId', 'score', 'features', 'shapBaseValue')


def build_output(
    keys: pl.DataFrame,
    matrix: np.ndarray,
    shap_values: np.ndarray | None,
    base_value: float | None,
    features: Sequence[str],
) -> pl.DataFrame:
    """Assemble the release schema from the scored keys and their explanations.

    The `features` array carries one struct per feature, in fitted order. Types are pinned here
    rather than left to inference: `gentropy_l2g_evidence` reads this dataset with an imposed
    schema, so a drift would be coerced silently instead of raised.

    Args:
        keys: `studyLocusId`, `geneId` and `score`, one row per prediction.
        matrix: the feature values behind those rows, in fitted order.
        shap_values: SHAP values with the shape of `matrix`, or None when explanations are off,
            in which case every `shapValue` is null.
        base_value: the model's expected value, or None when explanations are off.
        features: feature names, in fitted order.

    Returns:
        A DataFrame of `OUTPUT_COLUMNS`.
    """
    values = pl.DataFrame(matrix, schema=[(name, pl.Float32) for name in features])
    shap_schema = [(f'shap_{name}', pl.Float32) for name in features]
    shaps = (
        pl.DataFrame(shap_values, schema=shap_schema)
        if shap_values is not None
        else pl.DataFrame(
            {name: [None] * values.height for name, _ in shap_schema}, schema=shap_schema
        )
    )
    wide = pl.concat([keys, values, shaps], how='horizontal')

    return wide.select(
        pl.col('studyLocusId').cast(pl.String),
        pl.col('geneId').cast(pl.String),
        pl.col('score').cast(pl.Float64),
        pl.concat_list(
            pl.struct(
                pl.lit(name, dtype=pl.String).alias('name'),
                pl.col(name).cast(pl.Float32).alias('value'),
                pl.col(f'shap_{name}').cast(pl.Float32).alias('shapValue'),
            )
            for name in features
        ).alias('features'),
        pl.lit(base_value, dtype=pl.Float32).alias('shapBaseValue'),
    )


def l2g_predict(
    source: dict[str, str],
    destination: str,
    settings: dict[str, Any],
    config: Config,
) -> None:
    """Score the feature matrix, explain the survivors and write the release dataset.

    Args:
        source: keys `feature_matrix`, `credible_set`, `model`, `background`.
        destination: the output dataset directory.
        settings: keys `features_list`, `l2g_threshold`, `explain_predictions`,
            `shap_background_size`, and optionally `shap_workers`.
        config: otter config; unused, accepted for interface compatibility.
    """
    features = list(settings.get('features_list') or FEATURES)
    threshold = float(settings['l2g_threshold'])

    gwas_loci = (
        scan_dataset(source['credible_set'])
        .filter(pl.col('studyType') == 'gwas')
        .select('studyLocusId')
        .unique()
    )

    logger.info('preparing the prediction matrix')
    prepared = impute_and_cast(
        scan_dataset(source['feature_matrix'])
        .filter(pl.col('isProteinCoding') == 1)
        .join(gwas_loci, on='studyLocusId', how='semi'),
        features,
    ).collect()

    model = l2g_model.load_model(source['model'])
    matrix = l2g_model.to_matrix(prepared, features)
    logger.info(f'scoring {matrix.shape[0]} rows')
    scores = model.predict_proba(matrix)[:, 1] if matrix.shape[0] else np.empty(0, dtype=np.float32)

    keep = scores >= threshold
    keys = prepared.select('studyLocusId', 'geneId').filter(pl.Series(keep)).with_columns(
        pl.Series('score', scores[keep])
    )
    matrix = matrix[keep]
    logger.info(f'{keys.height} rows at or above the {threshold} threshold')

    if settings.get('explain_predictions'):
        background = pl.read_parquet(source['background']).select(features).to_numpy().astype(np.float32)
        workers = settings.get('shap_workers') or os.cpu_count()
        logger.info(f'explaining {matrix.shape[0]} rows on {workers} workers')
        base_value, shap_values = l2g_explain.explain(
            model,
            matrix,
            background,
            max_samples=int(settings['shap_background_size']),
            workers=workers,
        )
    else:
        # Null, not NaN. The flag says no explanation was computed, which is a different
        # statement from "the explanation is not a number", and consumers read them differently.
        base_value, shap_values = None, None

    output = build_output(keys, matrix, shap_values, base_value, features).sort(
        'studyLocusId', 'geneId'
    )
    write_dataset(output, destination)
    logger.info('prediction complete')
