"""Quality checks of credible set variants; the first credible set stage, which only adds flags."""

from __future__ import annotations

from typing import Any

import polars as pl
from loguru import logger
from otter.config.model import Config
from pydantic import BaseModel, ConfigDict, Field

from pts.schemas.credible_set import credible_set_variant_schema
from pts.transformers.credible_set.enums import Chromosome
from pts.transformers.credible_set.flags import CredibleSetVariantFlag
from pts.transformers.credible_set.variant_id import variant_alternate_allele, variant_reference_allele
from pts.transformers.utils.dataset import scan_dataset, sink_dataset
from pts.transformers.utils.quality_flags import flag_rows

#: A credible set identifier: a 32-character lowercase md5 hex digest.
CREDIBLE_SET_ID_PATTERN = r'^[0-9a-f]{32}$'

#: A variant identifier on a canonical chromosome, with a positive position and nucleotide alleles.
VARIANT_ID_PATTERN = rf'^(?:{"|".join(chromosome.value for chromosome in Chromosome)})_[1-9][0-9]*_[ACGT]+_[ACGT]+$'


class VariantQcSource(BaseModel):
    """The credible set variants of one source."""

    model_config = ConfigDict(extra='forbid')

    credible_set_variant: str


class VariantQcSettings(BaseModel):
    """Settings of the coverage and PIP checks."""

    model_config = ConfigDict(extra='forbid')

    coverage: float = Field(default=0.95, gt=0, le=1)
    """The cumulative PIP up to which variants belong to the credible set."""
    pip_tolerance: float = Field(default=1e-4, ge=0)
    """How far the PIP sum of the kept variants may lie outside [`coverage`, 1]."""


def _pip() -> pl.Expr:
    """The PIP as a double, so sums of single-precision PIPs keep their precision."""
    return pl.col('posteriorProbability').cast(pl.Float64)


def flag_invalid_cs_id(frame: pl.LazyFrame) -> pl.LazyFrame:
    """Flag credible set identifiers that are not a 32-character lowercase hexadecimal md5.

    Examples:
        >>> credible_set_variant = pl.LazyFrame({
        ...     'credibleSetId': ['0' * 32, 'A' * 32, '0' * 31],
        ...     'qualityControls': [[]] * 3,
        ... })
        >>> flag_invalid_cs_id(credible_set_variant).collect()['qualityControls'].list.len().to_list()
        [0, 1, 1]
    """
    invalid = ~pl.col('credibleSetId').str.contains(CREDIBLE_SET_ID_PATTERN)
    return flag_rows(frame, invalid, CredibleSetVariantFlag.INVALID_CS_ID)


def flag_invalid_variant_id(frame: pl.LazyFrame) -> pl.LazyFrame:
    """Flag variant identifiers that are not `chromosome_position_ref_alt`, or whose ref equals alt.

    A valid identifier has a chromosome of `Chromosome`, a position from 1, and reference and
    alternative alleles of the nucleotides A, C, G and T that differ.

    Examples:
        >>> credible_set_variant = pl.LazyFrame({
        ...     'variantId': [
        ...         '1_100_A_T', '22_100_A_T', 'XY_5_ACGT_A', '1_100_A_A', '23_100_A_T', '14_KI270846v1_alt_497149_C_T'
        ...     ],
        ...     'qualityControls': [[]] * 6,
        ... })
        >>> flagged = flag_invalid_variant_id(credible_set_variant).collect()
        >>> flagged.filter(pl.col('qualityControls').list.len() > 0)['variantId'].to_list()
        ['1_100_A_A', '23_100_A_T', '14_KI270846v1_alt_497149_C_T']
    """
    variant_id = pl.col('variantId')
    same_allele = variant_reference_allele(variant_id) == variant_alternate_allele(variant_id)
    invalid = ~variant_id.str.contains(VARIANT_ID_PATTERN) | same_allele
    return flag_rows(frame, invalid, CredibleSetVariantFlag.INVALID_VARIANT_ID)


def flag_exceeds_coverage(frame: pl.LazyFrame, coverage: float) -> pl.LazyFrame:
    """Flag the variants beyond the `coverage` of their credible set.

    Within a credible set, variants are ordered by PIP, highest first, then by `variantId`, so equal
    PIPs are cut the same way on every run. A variant exceeds the coverage when the PIP sum of the
    variants before it has reached `coverage`; the variant that brings the sum to `coverage` is kept.

    Examples:
        >>> credible_set_variant = pl.LazyFrame({
        ...     'credibleSetId': ['a'] * 4,
        ...     'variantId': ['1_1_A_T', '1_2_A_T', '1_3_A_T', '1_4_A_T'],
        ...     'posteriorProbability': [0.60, 0.30, 0.09, 0.01],
        ...     'qualityControls': [[]] * 4,
        ... })
        >>> flag_exceeds_coverage(credible_set_variant, 0.95).collect()['qualityControls'].list.len().to_list()
        [0, 0, 0, 1]
    """
    pip = _pip()
    before = pip.cum_sum().over('credibleSetId', order_by=[-pip, pl.col('variantId')]) - pip
    return flag_rows(frame, before >= coverage, CredibleSetVariantFlag.EXCEEDS_COVERAGE_95)


def flag_abnormal_pip(frame: pl.LazyFrame, coverage: float, pip_tolerance: float) -> pl.LazyFrame:
    """Flag every variant of a credible set whose PIPs are abnormal.

    A set is abnormal when the PIP sum of its variants without `EXCEEDS_COVERAGE_95` lies outside
    [`coverage` - `pip_tolerance`, 1 + `pip_tolerance`], or when any of its PIPs is missing, NaN,
    at most 0 or above 1. It runs after `flag_exceeds_coverage`.

    Examples:
        >>> credible_set_variant = pl.LazyFrame({
        ...     'credibleSetId': ['a', 'a', 'b', 'b'],
        ...     'variantId': ['1_1_A_T', '1_2_A_T', '1_3_A_T', '1_4_A_T'],
        ...     'posteriorProbability': [0.50, 0.30, 0.97, 0.02],
        ...     'qualityControls': [[]] * 4,
        ... })
        >>> flagged = credible_set_variant.pipe(flag_exceeds_coverage, 0.95).pipe(flag_abnormal_pip, 0.95, 1e-4)
        >>> flagged = flagged.collect()
        >>> flagged['qualityControls'].list.contains(CredibleSetVariantFlag.ABNORMAL_PIP.value).to_list()
        [True, True, False, False]
    """
    pip = _pip()
    exceeds = pl.col('qualityControls').list.contains(CredibleSetVariantFlag.EXCEEDS_COVERAGE_95.value)
    kept_sum = pip.filter(~exceeds).sum().over('credibleSetId')
    invalid_pip = (pip.is_null() | pip.is_nan() | (pip <= 0) | (pip > 1)).any().over('credibleSetId')
    abnormal = invalid_pip | ~kept_sum.is_between(coverage - pip_tolerance, 1 + pip_tolerance)
    return flag_rows(frame, abnormal, CredibleSetVariantFlag.ABNORMAL_PIP)


def flag_effect_direction_mismatch(frame: pl.LazyFrame) -> pl.LazyFrame:
    """Flag variants whose reported and conditional effects are both present and have opposite signs.

    A zero effect has no sign, so it never mismatches.

    Examples:
        >>> credible_set_variant = pl.LazyFrame({
        ...     'reportedBeta': [0.2, 0.2, 0.2, 0.0],
        ...     'conditionalBeta': [-0.1, 0.1, None, -0.1],
        ...     'qualityControls': [[]] * 4,
        ... })
        >>> flag_effect_direction_mismatch(credible_set_variant).collect()['qualityControls'].list.len().to_list()
        [1, 0, 0, 0]
    """
    opposite = pl.col('reportedBeta').sign() * pl.col('conditionalBeta').sign() == -1
    return flag_rows(frame, opposite, CredibleSetVariantFlag.EFFECT_DIRECTION_MISMATCH)


def flag_incomplete_stats(frame: pl.LazyFrame) -> pl.LazyFrame:
    """Flag variants with exactly one of the beta and standard error, for the reported and the conditional effect.

    Both missing is not incomplete: the effect is simply not reported.

    Examples:
        >>> credible_set_variant = pl.LazyFrame({
        ...     'reportedBeta': [0.2, None, 0.2, None],
        ...     'reportedStandardError': [None, 0.01, 0.01, None],
        ...     'conditionalBeta': [None] * 4,
        ...     'conditionalStandardError': [None] * 4,
        ...     'qualityControls': [[]] * 4,
        ... }, schema_overrides={'conditionalBeta': pl.Float32, 'conditionalStandardError': pl.Float32})
        >>> flag_incomplete_stats(credible_set_variant).collect()['qualityControls'].list.len().to_list()
        [1, 1, 0, 0]
    """

    def exactly_one_missing(beta: str, standard_error: str) -> pl.Expr:
        return pl.col(beta).is_null() != pl.col(standard_error).is_null()

    return frame.pipe(
        flag_rows,
        exactly_one_missing('reportedBeta', 'reportedStandardError'),
        CredibleSetVariantFlag.INCOMPLETE_REPORTED_STATS,
    ).pipe(
        flag_rows,
        exactly_one_missing('conditionalBeta', 'conditionalStandardError'),
        CredibleSetVariantFlag.INCOMPLETE_CONDITIONAL_STATS,
    )


def flag_missing_pvalue(frame: pl.LazyFrame) -> pl.LazyFrame:
    """Flag variants whose p-value mantissa or exponent is missing."""
    missing = pl.col('pValueMantissa').is_null() | pl.col('pValueExponent').is_null()
    return flag_rows(frame, missing, CredibleSetVariantFlag.MISSING_PVALUE)


def flag_missing_sample_size(frame: pl.LazyFrame) -> pl.LazyFrame:
    """Flag variants without a sample size; the rescaling falls back to the study sample size."""
    return flag_rows(frame, pl.col('sampleSize').is_null(), CredibleSetVariantFlag.MISSING_SAMPLE_SIZE)


def flag_variants(frame: pl.LazyFrame, coverage: float, pip_tolerance: float) -> pl.LazyFrame:
    """Add every variant QC flag; the coverage cut runs before the PIP check, which uses it."""
    return (
        frame
        .pipe(flag_invalid_cs_id)
        .pipe(flag_invalid_variant_id)
        .pipe(flag_exceeds_coverage, coverage)
        .pipe(flag_abnormal_pip, coverage, pip_tolerance)
        .pipe(flag_effect_direction_mismatch)
        .pipe(flag_incomplete_stats)
        .pipe(flag_missing_pvalue)
        .pipe(flag_missing_sample_size)
    )


def variant_qc(
    source: dict[str, str | list[str]],
    destination: str,
    settings: dict[str, Any],
    config: Config,
) -> None:
    """Add quality flags to the credible set variants of one source.

    Args:
        source: The source's CredibleSetVariant path under `credible_set_variant`.
        destination: Path of the flagged variants.
        settings: `coverage` and `pip_tolerance` of the coverage and PIP checks.
        config: Pipeline configuration.
    """
    sources = VariantQcSource.model_validate(source)
    options = VariantQcSettings.model_validate(settings)
    flagged = (
        scan_dataset(sources.credible_set_variant)
        .match_to_schema(credible_set_variant_schema)
        .pipe(flag_variants, options.coverage, options.pip_tolerance)
    )
    logger.info('Writing flagged credible set variants to {}', destination)
    sink_dataset(flagged, destination).collect()
