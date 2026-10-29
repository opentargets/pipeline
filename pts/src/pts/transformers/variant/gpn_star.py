"""GPN-Star variant effect scores: cut-offs, normalisation and assessment."""

from __future__ import annotations

import math
from typing import Self

import polars as pl
from pydantic import BaseModel, ConfigDict, model_validator

from pts.schemas.variant_annotation import variant_effect_dtype
from pts.transformers.utils.dataset import list_parts
from pts.transformers.utils.variant import min_max_scaler

METHOD = 'GPN-Star'
FILE_PATTERN = 'llr_chr*.parquet'
KEY = ['position', 'referenceAllele', 'alternateAllele']


class GpnStarCutoffs(BaseModel):
    """Calibrated log-likelihood-ratio cut-offs; the more negative the score, the more constrained."""

    model_config = ConfigDict(extra='forbid')

    top_10_percent: float
    """Score at the edge of the 10% most constrained variants."""
    top_1_percent: float
    """Score at the edge of the 1% most constrained variants."""
    top_0_1_percent: float
    """Score at the edge of the 0.1% most constrained variants."""
    minimum: float
    """Score at and below which the normalised score saturates at 1."""

    @model_validator(mode='after')
    def _check_order(self) -> Self:
        if not self.minimum < self.top_0_1_percent < self.top_1_percent < self.top_10_percent:
            raise ValueError('cut-offs must satisfy minimum < top_0_1_percent < top_1_percent < top_10_percent')
        return self


def normalise_gpn_star(llr: pl.Expr, cutoffs: GpnStarCutoffs) -> pl.Expr:
    """Map the calibrated score onto [0, 1], higher meaning more constrained."""
    value = llr.cast(pl.Float64)
    return (
        pl
        .when(value >= cutoffs.top_10_percent)
        .then(0.0)
        .when(value >= cutoffs.top_1_percent)
        .then(min_max_scaler(value, cutoffs.top_10_percent, cutoffs.top_1_percent, 0.0, 0.5))
        .when(value >= cutoffs.top_0_1_percent)
        .then(min_max_scaler(value, cutoffs.top_1_percent, cutoffs.top_0_1_percent, 0.5, 0.75))
        .when(value.is_not_null())
        .then(min_max_scaler(value, cutoffs.top_0_1_percent, cutoffs.minimum, 0.75, 1.0).clip(upper_bound=1.0))
    )


def assess_gpn_star(llr: pl.Expr, cutoffs: GpnStarCutoffs) -> pl.Expr:
    """Label the score by the constraint bin it falls into."""
    return (
        pl
        .when(llr >= cutoffs.top_10_percent)
        .then(pl.lit('not constrained'))
        .when(llr >= cutoffs.top_1_percent)
        .then(pl.lit('constrained (top 10%)'))
        .when(llr >= cutoffs.top_0_1_percent)
        .then(pl.lit('constrained (top 1%)'))
        .when(llr.is_not_null())
        .then(pl.lit('constrained (top 0.1%)'))
    )


def gpn_star_files(path: str) -> dict[str, str]:
    """The score file of each chromosome below the directory `path`."""
    return {file.rsplit('llr_chr', 1)[1].removesuffix('.parquet'): file for file in list_parts(path, FILE_PATTERN)}


def llr_histogram(files: list[str]) -> pl.DataFrame:
    """Count the scores of all `files`, rounded to three decimals, one file at a time."""
    counts = [
        pl
        .scan_parquet(file)
        .select(pl.col('llr_calibrated').cast(pl.Float64).round(3).alias('llr'))
        .filter(pl.col('llr').is_not_null() & pl.col('llr').is_not_nan())
        .group_by('llr')
        .agg(pl.len().cast(pl.UInt64).alias('count'))
        .collect(engine='streaming')
        for file in files
    ]
    return pl.concat(counts).group_by('llr').agg(pl.col('count').sum()).sort('llr')


def cutoffs_from_histogram(histogram: pl.DataFrame) -> GpnStarCutoffs:
    """The score at which the cumulative count first reaches each fraction of the total."""
    ordered = histogram.sort('llr').with_columns(pl.col('count').cum_sum().alias('cumulative'))
    total = ordered['cumulative'][-1]

    def cutoff(fraction: float) -> float:
        needed = math.ceil(round(fraction * total, 9))
        return ordered.filter(pl.col('cumulative') >= needed)['llr'][0]

    return GpnStarCutoffs(
        top_10_percent=cutoff(0.1),
        top_1_percent=cutoff(0.01),
        top_0_1_percent=cutoff(0.001),
        minimum=ordered['llr'][0],
    )


def read_gpn_star(file: str, positions: pl.Series, cutoffs: GpnStarCutoffs) -> pl.DataFrame:
    """The scores of `file` at `positions`, as one variant effect per position and alleles.

    Raises:
        ValueError: when a position and alleles occur more than once.
    """
    llr = pl.col('llr_calibrated')
    scores = (
        pl
        .scan_parquet(file)
        .select('pos', 'ref', 'alt', 'llr_calibrated')
        .filter(pl.col('pos').is_in(positions.cast(pl.Int64).implode()))
        .select(
            pl.col('pos').cast(pl.Int32).alias('position'),
            pl.col('ref').alias('referenceAllele'),
            pl.col('alt').alias('alternateAllele'),
            pl
            .struct(
                pl.lit(METHOD).alias('method'),
                assess_gpn_star(llr, cutoffs).alias('assessment'),
                llr.alias('score'),
                pl.lit(None, dtype=pl.String).alias('assessmentFlag'),
                pl.lit(None, dtype=pl.String).alias('targetId'),
                normalise_gpn_star(llr, cutoffs).alias('normalisedScore'),
            )
            .cast(variant_effect_dtype.inner)
            .alias('gpnStarEffect'),
        )
        .collect(engine='streaming')
    )
    if scores.select(pl.struct(KEY).is_duplicated().any()).item():
        raise ValueError(f'duplicated position and alleles in {file}')
    return scores


def add_gpn_star(frame: pl.DataFrame, scores: pl.DataFrame) -> pl.DataFrame:
    """Append the matching GPN-Star effect to the variant effects of each row of `frame`."""
    joined = frame.join(scores, on=KEY, how='left', maintain_order='left')
    effects = pl.col('variantEffect')
    added = pl.col('gpnStarEffect')
    return joined.select(
        *[column for column in frame.columns if column != 'variantEffect'],
        pl
        .when(added.is_not_null())
        .then(pl.concat_list(effects.fill_null([]), added))
        .otherwise(effects)
        .alias('variantEffect'),
    ).select(frame.columns)
