"""The coloc_pip_ecaviar colocalisation of overlapping credible set pairs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyspark.sql.functions as f

from pts.schemas.colocalisation import COLOC_PIP_ECAVIAR, colocalisation_schema
from pts.schemas.spark import to_spark_schema

if TYPE_CHECKING:
    from pyspark.sql import Column, DataFrame

_FLOOR = 1e-16
_KEYS = ('chromosome', 'leftLocusId', 'rightLocusId', 'rightStudyType')


def _floored(column: str) -> Column:
    return f.greatest(f.coalesce(f.col(column), f.lit(0.0)), f.lit(_FLOOR))


def colocalise(
    aligned: DataFrame,
    priorc1: float = 1e-4,
    priorc2: float = 1e-4,
    priorc12: float = 1e-5,
) -> DataFrame:
    """Colocalise each pair of credible sets from its aligned tags.

    Args:
        aligned: The output of `aligned_tags`.
        priorc1: Prior probability that a variant is causal for the left trait only.
        priorc2: Prior probability that a variant is causal for the right trait only.
        priorc12: Prior probability that a variant is causal for both traits.

    Returns:
        One row per pair with the columns of `colocalisation_schema`.
    """
    left = f.col('left_posteriorProbability')
    right = f.col('right_posteriorProbability')
    usable_ratio = (
        f.col('left_beta').isNotNull()
        & f.col('right_beta').isNotNull()
        & (f.col('left_beta') != 0)
        & (f.col('right_beta') != 0)
    )
    p1, p2, p12 = (f.lit(max(p, _FLOOR)) for p in (priorc1, priorc2, priorc12))
    h3_unnormalised = p1 * p2 * f.greatest(f.col('s1') * f.col('s2') - f.col('b'), f.lit(0.0))
    h4_unnormalised = p12 * f.col('b')
    total = h3_unnormalised + h4_unnormalised
    schema = to_spark_schema(colocalisation_schema)
    return (
        aligned
        .groupBy(*_KEYS)
        .agg(
            f.sum(_floored('left_posteriorProbability')).alias('s1'),
            f.sum(_floored('right_posteriorProbability')).alias('s2'),
            f.sum(_floored('left_posteriorProbability') * _floored('right_posteriorProbability')).alias('b'),
            f.sum(f.when(left.isNotNull() & right.isNotNull(), 1).otherwise(0)).alias('numberColocalisingVariants'),
            f.sum(left * right).alias('clpp'),
            f.avg(f.when(usable_ratio, f.signum(f.col('left_beta') / f.col('right_beta')))).alias(
                'betaRatioSignAverage'
            ),
        )
        .withColumn('colocalisationMethod', f.lit(COLOC_PIP_ECAVIAR))
        .withColumn('h3', h3_unnormalised / total)
        .withColumn('h4', h4_unnormalised / total)
        .select(*[f.col(field.name).cast(field.dataType) for field in schema])
    )
