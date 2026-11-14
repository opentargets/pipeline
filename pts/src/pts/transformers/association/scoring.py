"""Association scores per datasource."""

from __future__ import annotations

import numpy as np
import polars as pl

from pts.transformers.association.harmonic import MAX_HARMONIC_SUM, harmonic_sum

KEY = ['targetId', 'diseaseId', 'datasourceId', 'datatypeId']
MAX_SCORES = 50
YEAR_SPAN = '100000i'


def aggregate_by_datasource(evidence: pl.LazyFrame) -> pl.LazyFrame:
    """Score each target-disease-datasource pair per year from all evidence up to that year."""
    yearly = evidence.group_by([*KEY, 'year']).agg(
        pl.col('score').sort(descending=True).head(MAX_SCORES).alias('top'),
        pl.len().cast(pl.Int32).alias('yearlyEvidenceCount'),
    )
    return (
        yearly
        .sort([*KEY, 'year'])
        .rolling('year', period=YEAR_SPAN, group_by=KEY, closed='right')
        .agg(
            pl.col('top').explode().sort(descending=True).head(MAX_SCORES).alias('pool'),
            pl.col('yearlyEvidenceCount').last(),
        )
        .with_columns(
            harmonic_sum(pl.col('pool'))
            .map_batches(lambda s: np.divide(s, MAX_HARMONIC_SUM), return_dtype=pl.Float64)
            .alias('associationScore'),
            pl.lit('datasourceId').alias('aggregationType'),
            pl.col('datasourceId').alias('aggregationValue'),
        )
        .select(
            'targetId',
            'diseaseId',
            'aggregationType',
            'aggregationValue',
            'datatypeId',
            'year',
            'associationScore',
            'yearlyEvidenceCount',
        )
    )
