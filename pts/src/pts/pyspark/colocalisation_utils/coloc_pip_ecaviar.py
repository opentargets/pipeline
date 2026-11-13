"""The coloc_pip_ecaviar colocalisation of overlapping credible set pairs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyspark.sql.functions as f

from pts.pyspark.colocalisation_utils.overlap import overlapping_tags
from pts.schemas.colocalisation import COLOC_PIP_ECAVIAR, colocalisation_schema
from pts.schemas.spark import to_spark_schema

if TYPE_CHECKING:
    from pyspark.sql import Column, DataFrame

_FLOOR = 1e-16
_KEYS = ('chromosome', 'leftLocusId', 'rightLocusId', 'rightStudyType')


def _floored(column: str) -> Column:
    return f.greatest(f.coalesce(f.col(column), f.lit(0.0)), f.lit(_FLOOR))


def colocalise(
    tags: DataFrame,
    priorc1: float = 1e-4,
    priorc2: float = 1e-4,
    priorc12: float = 1e-5,
) -> DataFrame:
    """Colocalise each overlapping pair of credible sets from its shared tags and per-locus totals.

    A tag in one locus only adds a floored posterior probability of `_FLOOR` on the other side, so
    the sums over all aligned tags follow from sums over the shared tags and per-locus totals.
    Assumes a variant appears at most once in a locus.

    Args:
        tags: The output of `overlap.tags`.
        priorc1: Prior probability that a variant is causal for the left trait only.
        priorc2: Prior probability that a variant is causal for the right trait only.
        priorc12: Prior probability that a variant is causal for both traits.

    Returns:
        One row per pair with the columns of `colocalisation_schema`.
    """
    totals = tags.groupBy('locusId').agg(
        f.count('*').alias('n'),
        f.sum(_floored('posteriorProbability')).alias('t'),
    )
    shared = overlapping_tags(tags)
    left = f.col('left_posteriorProbability')
    right = f.col('right_posteriorProbability')
    per_pair = shared.groupBy(*_KEYS).agg(
        f.count('*').alias('m'),
        f.sum(_floored('left_posteriorProbability')).alias('t1_shared'),
        f.sum(_floored('right_posteriorProbability')).alias('t2_shared'),
        f.sum(_floored('left_posteriorProbability') * _floored('right_posteriorProbability')).alias('b_shared'),
        f.sum(f.when(left.isNotNull() & right.isNotNull(), 1).otherwise(0)).alias('numberColocalisingVariants'),
        f.sum(left * right).alias('clpp'),
        f.avg(f.when(_usable_ratio(), f.signum(f.col('left_beta') / f.col('right_beta')))).alias(
            'betaRatioSignAverage'
        ),
    )
    floor = f.lit(_FLOOR)
    scored = (
        per_pair
        .join(
            totals.select(f.col('locusId').alias('leftLocusId'), f.col('n').alias('n1'), f.col('t').alias('t1')),
            on='leftLocusId',
        )
        .join(
            totals.select(f.col('locusId').alias('rightLocusId'), f.col('n').alias('n2'), f.col('t').alias('t2')),
            on='rightLocusId',
        )
        .withColumn('s1', f.col('t1') + (f.col('n2') - f.col('m')) * floor)
        .withColumn('s2', f.col('t2') + (f.col('n1') - f.col('m')) * floor)
        .withColumn(
            'b',
            f.col('b_shared') + (f.col('t1') - f.col('t1_shared')) * floor + (f.col('t2') - f.col('t2_shared')) * floor,
        )
    )
    return _posteriors(scored, priorc1, priorc2, priorc12)


def _usable_ratio() -> Column:
    return (
        f.col('left_beta').isNotNull()
        & f.col('right_beta').isNotNull()
        & (f.col('left_beta') != 0)
        & (f.col('right_beta') != 0)
    )


def _posteriors(per_pair: DataFrame, priorc1: float, priorc2: float, priorc12: float) -> DataFrame:
    p1, p2, p12 = (f.lit(max(p, _FLOOR)) for p in (priorc1, priorc2, priorc12))
    h3_unnormalised = p1 * p2 * f.greatest(f.col('s1') * f.col('s2') - f.col('b'), f.lit(0.0))
    h4_unnormalised = p12 * f.col('b')
    total = h3_unnormalised + h4_unnormalised
    schema = to_spark_schema(colocalisation_schema)
    return (
        per_pair
        .withColumn('colocalisationMethod', f.lit(COLOC_PIP_ECAVIAR))
        .withColumn('h3', h3_unnormalised / total)
        .withColumn('h4', h4_unnormalised / total)
        .select(*[f.col(field.name).cast(field.dataType) for field in schema])
    )
