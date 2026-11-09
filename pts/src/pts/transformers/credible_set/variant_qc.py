"""Variant QC: the first credible set stage, which flags each credible set variant of one source.

Nothing leaves here; validation decides what the flags exclude. `variant_qc` reads the source's
variants under their contract and runs the checks in order:

- the identifiers: a credible set id that is not an md5, a variant id that is not
  `chromosome_position_ref_alt` on a canonical chromosome, a variant listed twice in its set;
- the coverage: the members past the 95% coverage of their set, and the sets whose PIPs do not
  sum to it (the cut runs first, as the PIP check uses it);
- the statistics: reported and conditional effects of opposite sign, a beta without its standard
  error or the reverse, a missing p-value.
"""

from __future__ import annotations

from typing import Any

import polars as pl
from loguru import logger
from otter.config.model import Config
from pydantic import BaseModel, ConfigDict, Field

from pts.schemas.credible_set import credible_set_variant_required, credible_set_variant_schema
from pts.transformers.credible_set.flags import CredibleSetVariantFlag
from pts.transformers.credible_set.identifier import (
    CREDIBLE_SET_ID_PATTERN,
    VARIANT_ID_PATTERN,
    variant_alternate_allele,
    variant_reference_allele,
)
from pts.transformers.credible_set.source import read_source_dataset
from pts.transformers.utils.dataset import sink_dataset
from pts.transformers.utils.quality_flags import flag_rows


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
        read_source_dataset(sources.credible_set_variant, credible_set_variant_schema, credible_set_variant_required)
        .pipe(_flag_invalid_cs_id)
        .pipe(_flag_invalid_variant_id)
        .pipe(_flag_duplicated_variant)
        .pipe(_flag_exceeds_coverage, options.coverage)
        .pipe(_flag_abnormal_pip, options.coverage, options.pip_tolerance)
        .pipe(_flag_effect_direction_mismatch)
        .pipe(_flag_incomplete_stats)
        .pipe(_flag_missing_pvalue)
    )
    logger.info('Writing flagged credible set variants to {}', destination)
    sink_dataset(flagged, destination).collect()


def _flag_invalid_cs_id(frame: pl.LazyFrame) -> pl.LazyFrame:
    """Flag credible set identifiers that are not a 32-character lowercase hexadecimal md5.

    Examples:
        >>> credible_set_variant = pl.LazyFrame({
        ...     'credibleSetId': ['0' * 32, 'A' * 32, '0' * 31],
        ...     'qualityControls': [[]] * 3,
        ... })
        >>> _flag_invalid_cs_id(credible_set_variant).collect()['qualityControls'].list.len().to_list()
        [0, 1, 1]
    """
    invalid = ~pl.col('credibleSetId').str.contains(CREDIBLE_SET_ID_PATTERN)
    return flag_rows(frame, invalid, CredibleSetVariantFlag.INVALID_CS_ID)


def _flag_invalid_variant_id(frame: pl.LazyFrame) -> pl.LazyFrame:
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
        >>> flagged = _flag_invalid_variant_id(credible_set_variant).collect()
        >>> flagged.filter(pl.col('qualityControls').list.len() > 0)['variantId'].to_list()
        ['XY_5_ACGT_A', '1_100_A_A', '23_100_A_T', '14_KI270846v1_alt_497149_C_T']
    """
    variant_id = pl.col('variantId')
    same_allele = variant_reference_allele(variant_id) == variant_alternate_allele(variant_id)
    invalid = ~variant_id.str.contains(VARIANT_ID_PATTERN) | same_allele
    return flag_rows(frame, invalid, CredibleSetVariantFlag.INVALID_VARIANT_ID)


def _flag_duplicated_variant(frame: pl.LazyFrame) -> pl.LazyFrame:
    """Flag every copy of a variant that appears more than once in its credible set.

    No copy can be told apart as the right one, so every copy is flagged.

    Examples:
        >>> credible_set_variant = pl.LazyFrame({
        ...     'credibleSetId': ['a', 'a', 'a', 'b'],
        ...     'variantId': ['1_1_A_T', '1_1_A_T', '1_2_A_T', '1_1_A_T'],
        ...     'qualityControls': [[]] * 4,
        ... })
        >>> _flag_duplicated_variant(credible_set_variant).collect()['qualityControls'].list.len().to_list()
        [1, 1, 0, 0]
    """
    duplicated = pl.len().over('credibleSetId', 'variantId') > 1
    return flag_rows(frame, duplicated, CredibleSetVariantFlag.DUPLICATED_VARIANT)


def _flag_exceeds_coverage(frame: pl.LazyFrame, coverage: float) -> pl.LazyFrame:
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
        >>> _flag_exceeds_coverage(credible_set_variant, 0.95).collect()['qualityControls'].list.len().to_list()
        [0, 0, 0, 1]
    """
    pip = _pip()
    before = pip.cum_sum().over('credibleSetId', order_by=[-pip, pl.col('variantId')]) - pip
    return flag_rows(frame, before >= coverage, CredibleSetVariantFlag.EXCEEDS_COVERAGE_95)


def _pip() -> pl.Expr:
    """The PIP as a double, so sums of single-precision PIPs keep their precision."""
    return pl.col('posteriorProbability').cast(pl.Float64)


def _flag_abnormal_pip(frame: pl.LazyFrame, coverage: float, pip_tolerance: float) -> pl.LazyFrame:
    """Flag every variant of a credible set whose PIPs are abnormal.

    A set is abnormal when the PIP sum of its variants without `EXCEEDS_COVERAGE_95` lies outside
    [`coverage` - `pip_tolerance`, 1 + `pip_tolerance`], or when any of its PIPs is missing, NaN,
    at most 0 or above 1. It runs after `_flag_exceeds_coverage`.

    Examples:
        >>> credible_set_variant = pl.LazyFrame({
        ...     'credibleSetId': ['a', 'a', 'b', 'b'],
        ...     'variantId': ['1_1_A_T', '1_2_A_T', '1_3_A_T', '1_4_A_T'],
        ...     'posteriorProbability': [0.50, 0.30, 0.97, 0.02],
        ...     'qualityControls': [[]] * 4,
        ... })
        >>> flagged = credible_set_variant.pipe(_flag_exceeds_coverage, 0.95).pipe(_flag_abnormal_pip, 0.95, 1e-4)
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


def _flag_effect_direction_mismatch(frame: pl.LazyFrame) -> pl.LazyFrame:
    """Flag variants whose reported and conditional effects are both present and have opposite signs.

    A zero effect has no sign, so it never mismatches.

    Examples:
        >>> credible_set_variant = pl.LazyFrame({
        ...     'reportedBeta': [0.2, 0.2, 0.2, 0.0],
        ...     'conditionalBeta': [-0.1, 0.1, None, -0.1],
        ...     'qualityControls': [[]] * 4,
        ... })
        >>> _flag_effect_direction_mismatch(credible_set_variant).collect()['qualityControls'].list.len().to_list()
        [1, 0, 0, 0]
    """
    opposite = pl.col('reportedBeta').sign() * pl.col('conditionalBeta').sign() == -1
    return flag_rows(frame, opposite, CredibleSetVariantFlag.EFFECT_DIRECTION_MISMATCH)


def _flag_incomplete_stats(frame: pl.LazyFrame) -> pl.LazyFrame:
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
        >>> _flag_incomplete_stats(credible_set_variant).collect()['qualityControls'].list.len().to_list()
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


def _flag_missing_pvalue(frame: pl.LazyFrame) -> pl.LazyFrame:
    """Flag variants whose p-value mantissa or exponent is missing."""
    missing = pl.col('pValueMantissa').is_null() | pl.col('pValueExponent').is_null()
    return flag_rows(frame, missing, CredibleSetVariantFlag.MISSING_PVALUE)
