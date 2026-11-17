"""Association novelty and the per-group timeseries."""

from __future__ import annotations

import polars as pl

GROUP = ['diseaseId', 'targetId', 'aggregationType', 'aggregationValue']
PAIR = ['targetId', 'diseaseId']
BACKFILL_YEARS = 5
NULL_VALUE = 'NA'


def compute_novelty(
    association: pl.LazyFrame,
    *,
    current_year: int,
    window: int,
    scale: float,
    shift: float,
    evidence_count_dtype: pl.DataType,
) -> pl.LazyFrame:
    """Back-fill yearly scores, derive novelty from score increases and collapse into one row per group."""
    scores = association.select(
        'diseaseId',
        'targetId',
        'aggregationType',
        pl.col('aggregationValue').fill_null(NULL_VALUE),
        'year',
        'associationScore',
        pl.col('yearlyEvidenceCount').cast(evidence_count_dtype),
    ).with_columns(
        pl.col('year').min().over(PAIR).alias('first_year'),
        pl.struct(GROUP).rank('dense').alias('gid'),
    )
    groups = scores.group_by('gid').agg(*(pl.col(c).first() for c in [*GROUP, 'first_year']))

    grid = groups.select(
        'gid',
        pl.int_ranges(pl.col('first_year') - BACKFILL_YEARS, current_year + 2, dtype=pl.Int32).alias('year'),
    ).explode('year')
    new_group = pl.col('gid').ne_missing(pl.col('gid').shift(1))
    filled = (
        grid
        .join(
            scores.select('gid', 'year', 'associationScore', 'yearlyEvidenceCount'),
            on=['gid', 'year'],
            how='left',
        )
        .sort('gid', 'year')
        .with_columns(
            pl.when(new_group).then(0.0).otherwise(pl.col('associationScore')).forward_fill().alias('associationScore')
        )
    )

    peaks = (
        filled
        .with_columns(
            pl
            .when(new_group)
            .then(None)
            .otherwise(pl.col('associationScore') - pl.col('associationScore').shift(1))
            .alias('peak')
        )
        .filter(pl.col('peak') > 0)
        .select('gid', pl.col('year').alias('peak_year'), pl.col('peak').alias('peak_value'))
    )
    decay = (
        peaks
        .join(pl.LazyFrame({'offset': pl.int_range(0, window + 1, dtype=pl.Int32, eager=True)}), how='cross')
        .select(
            'gid',
            (pl.col('peak_year') + pl.col('offset')).alias('year'),
            (pl.col('peak_value') / (1.0 + (scale * (pl.col('offset') - shift)).exp())).alias('novelty'),
        )
        .group_by('gid', 'year')
        .agg(pl.col('novelty').max())
    )

    undated = pl.col('year') > current_year
    return (
        filled
        .join(decay, on=['gid', 'year'], how='left', maintain_order='left')
        .with_columns(pl.col('novelty').fill_null(0.0))
        .with_columns(
            pl.when(undated).then(None).otherwise(pl.col('year')).alias('year'),
            pl.when(undated).then(None).otherwise(pl.col('novelty')).alias('novelty'),
        )
        .group_by('gid')
        .agg(
            pl.col('associationScore').filter(pl.col('year').is_null()).first().alias('undated_score'),
            pl.col('yearlyEvidenceCount').cast(pl.Int64).sum().alias('evidenceCount'),
            pl
            .struct('year', 'associationScore', 'novelty', 'yearlyEvidenceCount')
            .sort_by('year', nulls_last=True)
            .alias('timeseries'),
            pl.col('novelty').filter(pl.col('year') == current_year).first().alias('currentNovelty'),
        )
        .join(groups.select('gid', *GROUP), on='gid')
        .select(
            'diseaseId',
            'targetId',
            'aggregationType',
            pl
            .when(pl.col('aggregationValue') != NULL_VALUE)
            .then(pl.col('aggregationValue'))
            .alias('aggregationValue'),
            pl.col('undated_score').alias('associationScore'),
            'evidenceCount',
            'timeseries',
            'currentNovelty',
        )
    )
