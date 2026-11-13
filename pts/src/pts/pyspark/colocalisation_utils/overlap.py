"""Tags of credible sets and the shared tags of overlapping credible set pairs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyspark.sql.functions as f
from pyspark.sql import types as t

if TYPE_CHECKING:
    from pyspark.sql import DataFrame

_STATISTICS = ('posteriorProbability', 'beta')


def tags(credible_set: DataFrame, credible_set_variant: DataFrame) -> DataFrame:
    """Join each credible set variant to its set: one row per tag of a locus.

    Args:
        credible_set: Credible sets with `id`, `studyType` and `chromosome`.
        credible_set_variant: Credible set variants with `credibleSetId`, `variantId`,
            `posteriorProbability`, `conditionalBeta` and `reportedBeta`.

    Returns:
        Columns `locusId`, `studyType`, `chromosome`, `tagVariantId`, `posteriorProbability`
        (double) and `beta` (double, conditional where given, reported otherwise).
    """
    return credible_set_variant.join(
        credible_set.select(f.col('id').alias('credibleSetId'), 'studyType', 'chromosome'),
        on='credibleSetId',
    ).select(
        f.col('credibleSetId').alias('locusId'),
        'studyType',
        'chromosome',
        f.col('variantId').alias('tagVariantId'),
        f.col('posteriorProbability').cast(t.DoubleType()),
        f.coalesce(f.col('conditionalBeta').cast(t.DoubleType()), f.col('reportedBeta')).alias('beta'),
    )


def overlapping_tags(tags: DataFrame) -> DataFrame:
    """Join the tags of each GWAS locus to the tags of the loci it overlaps, on the shared variant.

    The left locus is a GWAS. A right locus is any molecular trait locus, or a GWAS locus with a
    smaller id.

    Args:
        tags: The output of `tags`.

    Returns:
        One row per pair and shared tag variant: `leftLocusId`, `rightLocusId`, `rightStudyType`,
        `chromosome`, `tagVariantId`, and `left_` and `right_` `posteriorProbability` and `beta`.
    """
    left = tags.alias('left')
    right = tags.alias('right')
    return left.join(
        right,
        on=[
            f.col('left.chromosome') == f.col('right.chromosome'),
            f.col('left.tagVariantId') == f.col('right.tagVariantId'),
            (f.col('right.studyType') != 'gwas') | (f.col('left.locusId') > f.col('right.locusId')),
            f.col('left.studyType') == f.lit('gwas'),
        ],
        how='inner',
    ).select(
        f.col('left.locusId').alias('leftLocusId'),
        f.col('right.locusId').alias('rightLocusId'),
        f.col('right.studyType').alias('rightStudyType'),
        f.col('left.chromosome').alias('chromosome'),
        f.col('left.tagVariantId').alias('tagVariantId'),
        *[f.col(f'left.{c}').alias(f'left_{c}') for c in _STATISTICS],
        *[f.col(f'right.{c}').alias(f'right_{c}') for c in _STATISTICS],
    )
