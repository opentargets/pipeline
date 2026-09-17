"""The L2G feature contract, shared by training and prediction.

`FEATURES` is the order the model was fitted on AND the order of the `features` array in
`output/l2g_prediction`. It is a contract, not a preference: sorting it or deriving it from a
schema silently changes what the classifier is handed.
"""

from collections.abc import Sequence

import polars as pl

FEATURES: tuple[str, ...] = (
    'eQtlColocClppMaximum',
    'pQtlColocClppMaximum',
    'sQtlColocClppMaximum',
    'eQtlColocH4Maximum',
    'pQtlColocH4Maximum',
    'sQtlColocH4Maximum',
    'eQtlColocClppMaximumNeighbourhood',
    'pQtlColocClppMaximumNeighbourhood',
    'sQtlColocClppMaximumNeighbourhood',
    'eQtlColocH4MaximumNeighbourhood',
    'pQtlColocH4MaximumNeighbourhood',
    'sQtlColocH4MaximumNeighbourhood',
    'distanceSentinelFootprint',
    'distanceSentinelFootprintNeighbourhood',
    'distanceFootprintMean',
    'distanceFootprintMeanNeighbourhood',
    'distanceTssMean',
    'distanceTssMeanNeighbourhood',
    'distanceSentinelTss',
    'distanceSentinelTssNeighbourhood',
    'vepMaximum',
    'vepMaximumNeighbourhood',
    'vepMean',
    'vepMeanNeighbourhood',
    'e2gMean',
    'e2gMeanNeighbourhood',
    'geneCount500kb',
    'proteinGeneCount500kb',
    'credibleSetConfidence',
    'transPQtlColocH4Maximum',
    'transPQtlColocH4MaximumNeighbourhood',
)
"""The 31 features, in the order `LocusToGeneConfig.features_list` declares them."""

FIXED_COLUMNS: tuple[str, ...] = ('studyLocusId', 'geneId')
"""Identity columns that are never handed to the classifier."""

LOCUS_MEAN_IMPUTED: tuple[str, ...] = ('geneCount500kb', 'proteinGeneCount500kb')
"""Gene attributes imputed with the mean over their study locus before the zero fill."""


def impute_and_cast(
    frame: pl.LazyFrame,
    features: Sequence[str],
    *,
    keep: Sequence[str] = (),
) -> pl.LazyFrame:
    """Reproduce gentropy's `L2GFeatureMatrix.__init__` cast, `fill_na`, and `select_features`.

    Every requested feature is cast to `Float32` first, mirroring `L2GFeatureMatrix.__init__`,
    which casts before anything else touches the matrix. The two gene-count columns are then
    filled with the mean over their `studyLocusId` partition; every remaining null in every
    feature becomes 0.0. Features are returned in the requested order, which is the order the
    classifier expects.

    The locus mean is computed in `Float64` and cast back to `Float32`, because spark's
    `avg` returns a double and `select_features` casts the result back to float. Averaging in
    `Float32` throughout would round differently.

    Args:
        frame: feature matrix, one row per (study locus, gene).
        features: feature names, in the order the model was fitted on.
        keep: extra columns to carry through, placed after the fixed columns.

    Returns:
        LazyFrame of `FIXED_COLUMNS + keep + features`, features as `Float32`.

    Raises:
        ValueError: if any requested feature is absent from the frame.
    """
    available = set(frame.collect_schema().names())
    if missing := [name for name in features if name not in available]:
        msg = f'feature matrix is missing {missing}'
        raise ValueError(msg)

    cast = frame.with_columns(pl.col(name).cast(pl.Float32).alias(name) for name in features)

    imputed = cast.with_columns(
        pl.col(name)
        .fill_null(pl.col(name).cast(pl.Float64).mean().over('studyLocusId').cast(pl.Float32))
        .alias(name)
        for name in LOCUS_MEAN_IMPUTED
        if name in features
    )

    return imputed.select(
        *FIXED_COLUMNS,
        *keep,
        *(pl.col(name).fill_null(0.0).alias(name) for name in features),
    )
