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


def aggregate_overall(by_datasource: pl.LazyFrame, weights: pl.DataFrame) -> pl.LazyFrame:
    """Aggregate the weighted datasource scores into one overall score per target-disease-year."""
    return _aggregate(
        by_datasource,
        weights,
        aggregation_type='overall',
        aggregation_value=pl.lit('None'),
        score=pl.col('associationScore') * pl.col('weight'),
        association=harmonic_sum(pl.col('maxes')).map_batches(
            lambda s: np.divide(s, MAX_HARMONIC_SUM), return_dtype=pl.Float64
        ),
    )


def aggregate_by_datatype(by_datasource: pl.LazyFrame, weights: pl.DataFrame) -> pl.LazyFrame:
    """Aggregate the datasource scores per datatype, normalised by the best possible harmonic sum."""
    return _aggregate(
        by_datasource,
        weights,
        aggregation_type='datatypeId',
        aggregation_value=pl.col('datatypeId'),
        score=pl.col('associationScore'),
        association=harmonic_sum(pl.col('maxes')) / harmonic_sum(pl.col('maxes').list.eval(pl.element() * 0.0 + 1.0)),
    )


def _aggregate(
    by_datasource: pl.LazyFrame,
    weights: pl.DataFrame,
    aggregation_type: str,
    aggregation_value: pl.Expr,
    score: pl.Expr,
    association: pl.Expr,
) -> pl.LazyFrame:
    part = ['targetId', 'diseaseId', 'aggregationValue']
    rows = (
        by_datasource
        .select(
            'targetId',
            'diseaseId',
            'year',
            'associationScore',
            'yearlyEvidenceCount',
            pl.col('aggregationValue').alias('datasourceId'),
        )
        .join(weights.lazy().select('datasourceId', 'weight', 'datatypeId'), on='datasourceId', how='inner')
        .select(
            'targetId',
            'diseaseId',
            aggregation_value.alias('aggregationValue'),
            'datasourceId',
            'year',
            score.alias('associationScore'),
            'yearlyEvidenceCount',
        )
        .sort([*part, 'year', 'datasourceId'])
        .with_columns(pl.int_range(pl.len()).over(part).alias('idx'))
    )
    states = (
        rows
        .with_columns(pl.col('associationScore').cum_max().over([*part, 'datasourceId']).alias('best'))
        .select(*part, 'datasourceId', 'idx', 'best')
        .sort('idx')
    )
    probes = (
        rows
        .select(*part, 'idx', 'year', 'yearlyEvidenceCount')
        .join(rows.select(*part, 'datasourceId').unique(), on=part)
        .sort('idx')
    )
    return (
        probes
        .join_asof(states, on='idx', by=[*part, 'datasourceId'], strategy='backward')
        .drop_nulls('best')
        .group_by([*part, 'idx', 'year', 'yearlyEvidenceCount'])
        .agg(pl.col('best').alias('maxes'))
        .with_columns(association.alias('associationScore'))
        .group_by([*part, 'year'])
        .agg(
            pl.col('associationScore').max(),
            pl.col('yearlyEvidenceCount').sum().cast(pl.Int64),
        )
        .with_columns(pl.lit(aggregation_type).alias('aggregationType'))
        .select(
            'targetId',
            'diseaseId',
            'aggregationType',
            'aggregationValue',
            'year',
            'associationScore',
            'yearlyEvidenceCount',
        )
    )
