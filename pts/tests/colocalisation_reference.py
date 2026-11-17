"""Reference coloc_pip_ecaviar over all aligned tags of each pair, as gentropy computes it.

Slow: it materialises every tag of both loci of every overlapping pair. Used only to check
`coloc_pip_ecaviar.colocalise` against.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyspark.sql.functions as f

from pts.schemas.colocalisation import COLOC_PIP_ECAVIAR, colocalisation_schema
from pts.schemas.spark import to_spark_schema

if TYPE_CHECKING:
    from pyspark.sql import Column, DataFrame

_FLOOR = 1e-16
_KEYS = ('chromosome', 'leftLocusId', 'rightLocusId', 'rightStudyType')
_STATISTICS = ('posteriorProbability', 'beta')


def _floored(column: str) -> Column:
    return f.greatest(f.coalesce(f.col(column), f.lit(0.0)), f.lit(_FLOOR))


def overlapping_pairs(tags: DataFrame) -> DataFrame:
    """Distinct pairs of loci sharing a tag: a GWAS on the left, a QTL or a smaller GWAS id on the right."""
    minimal = tags.select('locusId', 'studyType', 'chromosome', 'tagVariantId')
    return (
        minimal
        .alias('left')
        .join(
            minimal.alias('right'),
            on=[
                f.col('left.chromosome') == f.col('right.chromosome'),
                f.col('left.tagVariantId') == f.col('right.tagVariantId'),
                (f.col('right.studyType') != 'gwas') | (f.col('left.locusId') > f.col('right.locusId')),
                f.col('left.studyType') == f.lit('gwas'),
            ],
        )
        .select(
            f.col('left.locusId').alias('leftLocusId'),
            f.col('right.locusId').alias('rightLocusId'),
            f.col('right.studyType').alias('rightStudyType'),
            f.col('left.chromosome').alias('chromosome'),
        )
        .distinct()
    )


def aligned_tags(tags: DataFrame, pairs: DataFrame) -> DataFrame:
    """All tags of both loci of each pair, outer-joined on the tag variant."""
    left = tags.select(
        'chromosome',
        'tagVariantId',
        f.col('locusId').alias('leftLocusId'),
        *[f.col(c).alias(f'left_{c}') for c in _STATISTICS],
    ).join(pairs, on=['chromosome', 'leftLocusId'])
    right = tags.select(
        'chromosome',
        'tagVariantId',
        f.col('locusId').alias('rightLocusId'),
        *[f.col(c).alias(f'right_{c}') for c in _STATISTICS],
    ).join(pairs, on=['chromosome', 'rightLocusId'])
    return left.join(
        right,
        on=['chromosome', 'rightLocusId', 'leftLocusId', 'tagVariantId', 'rightStudyType'],
        how='outer',
    )


def colocalise(tags: DataFrame, priorc1: float = 1e-4, priorc2: float = 1e-4, priorc12: float = 1e-5) -> DataFrame:
    """Score every overlapping pair from its aligned tags."""
    aligned = aligned_tags(tags, overlapping_pairs(tags))
    left = f.col('left_posteriorProbability')
    right = f.col('right_posteriorProbability')
    usable = (
        f.col('left_beta').isNotNull()
        & f.col('right_beta').isNotNull()
        & (f.col('left_beta') != 0)
        & (f.col('right_beta') != 0)
    )
    per_pair = aligned.groupBy(*_KEYS).agg(
        f.sum(_floored('left_posteriorProbability')).alias('s1'),
        f.sum(_floored('right_posteriorProbability')).alias('s2'),
        f.sum(_floored('left_posteriorProbability') * _floored('right_posteriorProbability')).alias('b'),
        f.sum(f.when(left.isNotNull() & right.isNotNull(), 1).otherwise(0)).alias('numberColocalisingVariants'),
        f.sum(left * right).alias('clpp'),
        f.avg(f.when(usable, f.signum(f.col('left_beta') / f.col('right_beta')))).alias('betaRatioSignAverage'),
    )
    p1, p2, p12 = (f.lit(max(p, _FLOOR)) for p in (priorc1, priorc2, priorc12))
    h3 = p1 * p2 * f.greatest(f.col('s1') * f.col('s2') - f.col('b'), f.lit(0.0))
    h4 = p12 * f.col('b')
    schema = to_spark_schema(colocalisation_schema)
    return (
        per_pair
        .withColumn('colocalisationMethod', f.lit(COLOC_PIP_ECAVIAR))
        .withColumn('h3', h3 / (h3 + h4))
        .withColumn('h4', h4 / (h3 + h4))
        .select(*[f.col(field.name).cast(field.dataType) for field in schema])
    )
