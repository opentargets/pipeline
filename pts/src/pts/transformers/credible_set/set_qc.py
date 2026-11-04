"""Quality checks of credible sets; the second credible set stage, which only adds flags."""

from typing import Any

import polars as pl
from loguru import logger
from otter.config.model import Config
from pydantic import BaseModel, ConfigDict, Field

from pts.schemas.credible_set import credible_set_schema
from pts.schemas.study import StudyType
from pts.transformers.credible_set.enums import Chromosome, FinemappingMethod, KnownGenomicRegion
from pts.transformers.credible_set.flags import CredibleSetFlag
from pts.transformers.credible_set.variant_qc import CREDIBLE_SET_ID_PATTERN
from pts.transformers.study import QualityFlag as StudyQualityFlag
from pts.transformers.utils.dataset import scan_dataset, sink_dataset
from pts.transformers.utils.quality_flags import flag_rows


class SetQcSource(BaseModel):
    """The credible sets and variants of one source, and the validated study index."""

    model_config = ConfigDict(extra='forbid')

    credible_set: str
    credible_set_variant: str
    study: str


class SetQcSettings(BaseModel):
    """Settings of set QC."""

    model_config = ConfigDict(extra='forbid')

    allowed_fm_methods: list[FinemappingMethod] = Field(default_factory=lambda: list(FinemappingMethod))
    """The accepted `fineMappingMethod` values; any other value gets INVALID_FINEMAPPING_METHOD."""


def flag_invalid_cs_id(frame: pl.LazyFrame) -> pl.LazyFrame:
    """Flag credible set identifiers that are not a 32-character lowercase hexadecimal md5.

    Examples:
        >>> sets = pl.LazyFrame({'id': ['0' * 32, 'A' * 32, '0' * 31], 'qualityControls': [[]] * 3})
        >>> flag_invalid_cs_id(sets).collect()['qualityControls'].list.len().to_list()
        [0, 1, 1]
    """
    invalid = ~pl.col('id').str.contains(CREDIBLE_SET_ID_PATTERN)
    return flag_rows(frame, invalid, CredibleSetFlag.INVALID_CS_ID)


def flag_invalid_chromosome(frame: pl.LazyFrame) -> pl.LazyFrame:
    """Flag credible sets whose `chromosome` is not one of `Chromosome` (1 to 22, X, Y, XY, MT).

    Examples:
        >>> sets = pl.LazyFrame({'chromosome': ['1', 'XY', '23', 'chr1'], 'qualityControls': [[]] * 4})
        >>> flag_invalid_chromosome(sets).collect()['qualityControls'].list.len().to_list()
        [0, 0, 1, 1]
    """
    invalid = ~pl.col('chromosome').is_in([chromosome.value for chromosome in Chromosome])
    return flag_rows(frame, invalid, CredibleSetFlag.INVALID_CHROMOSOME)


def flag_lead_chromosome_mismatch(frame: pl.LazyFrame) -> pl.LazyFrame:
    """Flag credible sets whose lead variant lies on another chromosome than the set.

    The chromosome of the lead is the first field of `leadVariantId`.

    Examples:
        >>> sets = pl.LazyFrame({
        ...     'chromosome': ['1', '1', 'X'],
        ...     'leadVariantId': ['2_100_A_T', '1_100_A_T', '23_100_A_T'],
        ...     'qualityControls': [[]] * 3,
        ... })
        >>> flag_lead_chromosome_mismatch(sets).collect()['qualityControls'].list.len().to_list()
        [1, 0, 1]
    """
    mismatch = pl.col('leadVariantId').str.split('_').list.first() != pl.col('chromosome')
    return flag_rows(frame, mismatch, CredibleSetFlag.LEAD_CHROMOSOME_MISMATCH)


def flag_invalid_finemapping_method(frame: pl.LazyFrame, allowed: list[FinemappingMethod]) -> pl.LazyFrame:
    """Flag credible sets whose `fineMappingMethod` is not one of `allowed`; the comparison is exact.

    Examples:
        >>> sets = pl.LazyFrame({'fineMappingMethod': ['SuSiE', 'SuSie', 'PICS'], 'qualityControls': [[]] * 3})
        >>> flagged = flag_invalid_finemapping_method(sets, [FinemappingMethod.SUSIE])
        >>> flagged.collect()['qualityControls'].list.len().to_list()
        [0, 1, 1]
    """
    invalid = ~pl.col('fineMappingMethod').is_in([method.value for method in allowed])
    return flag_rows(frame, invalid, CredibleSetFlag.INVALID_FINEMAPPING_METHOD)


def lead_position() -> pl.Expr:
    """The position of the lead variant, the second field of `leadVariantId`."""
    return pl.col('leadVariantId').str.split('_').list.get(1).cast(pl.Int64, strict=False)


def flag_in_mhc(frame: pl.LazyFrame) -> pl.LazyFrame:
    """Flag credible sets whose lead variant lies in the MHC region.

    The lead position is the second field of `leadVariantId`, and the region bounds of
    `KnownGenomicRegion.MHC` are inclusive.

    Examples:
        >>> sets = pl.LazyFrame({
        ...     'chromosome': ['6', '6', '6', '7'],
        ...     'leadVariantId': ['6_25726063_A_T', '6_33400556_A_T', '6_33400557_A_T', '7_30000000_A_T'],
        ...     'qualityControls': [[]] * 4,
        ... })
        >>> flag_in_mhc(sets).collect()['qualityControls'].list.len().to_list()
        [1, 1, 0, 0]
    """
    mhc = KnownGenomicRegion.MHC.value
    in_mhc = (pl.col('chromosome') == mhc.chromosome.value) & lead_position().is_between(mhc.start, mhc.end)
    return flag_rows(frame, in_mhc, CredibleSetFlag.IN_MHC)


def flag_study(frame: pl.LazyFrame, studies: pl.LazyFrame) -> pl.LazyFrame:
    """Flag credible sets against the study index; the credible set keeps its own `studyType`.

    - INVALID_STUDY_TYPE: the source `studyType` is not one of `StudyType`; a missing type is not.
    - STUDY_TYPE_MISMATCH: the study is in the index, and the `studyType` of the set is missing or
      differs from the study's.
    - FLAGGED_STUDY: the study carries at least one quality control flag.
    - TOP_HIT_AND_SUMMARY_STATS: the set is a curated top hit, and the study is not flagged
      SUMSTATS_NOT_AVAILABLE, so its summary statistics should give the set instead.
    - MISSING_STUDY: the study is not in the study index.

    Examples:
        >>> sets = pl.LazyFrame({
        ...     'studyId': ['s', 't', 'unknown'],
        ...     'studyType': ['eQTL', 'gwas', 'gwas'],
        ...     'qualityControls': [[], [], []],
        ... })
        >>> studies = pl.LazyFrame({
        ...     'studyId': ['s', 't'],
        ...     'studyType': ['eqtl', 'gwas'],
        ...     'qualityControls': [[], []],
        ... })
        >>> for row in flag_study(sets, studies).sort('studyId').collect().iter_rows(named=True):
        ...     flags = sorted(CredibleSetFlag(flag).name for flag in row['qualityControls'])
        ...     print(row['studyId'], row['studyType'], flags)
        s eQTL ['INVALID_STUDY_TYPE', 'STUDY_TYPE_MISMATCH']
        t gwas []
        unknown gwas ['MISSING_STUDY']
    """
    lookup = studies.select(
        'studyId',
        pl.col('studyType').alias('_study_type'),
        pl.col('qualityControls').alias('_study_qc'),
        pl.lit(True).alias('_study_found'),
    )
    found = pl.col('_study_found').is_not_null()
    mismatch = found & (pl.col('studyType').is_null() | (pl.col('studyType') != pl.col('_study_type')))
    top_hit = pl.col('qualityControls').list.contains(CredibleSetFlag.TOP_HIT.value)
    no_sumstats = pl.col('_study_qc').list.contains(StudyQualityFlag.SUMSTATS_NOT_AVAILABLE.value)
    return (
        frame
        .join(lookup, on='studyId', how='left')
        .pipe(flag_rows, ~pl.col('studyType').is_in([t.value for t in StudyType]), CredibleSetFlag.INVALID_STUDY_TYPE)
        .pipe(flag_rows, mismatch, CredibleSetFlag.STUDY_TYPE_MISMATCH)
        .pipe(flag_rows, pl.col('_study_qc').list.len() > 0, CredibleSetFlag.FLAGGED_STUDY)
        .pipe(flag_rows, top_hit & ~no_sumstats, CredibleSetFlag.TOP_HIT_AND_SUMMARY_STATS)
        .pipe(flag_rows, ~found, CredibleSetFlag.MISSING_STUDY)
        .drop('_study_type', '_study_qc', '_study_found')
    )


def flag_lead_variant_missing(frame: pl.LazyFrame, variants: pl.LazyFrame) -> pl.LazyFrame:
    """Flag credible sets whose lead variant has no row among the variants of their source.

    Examples:
        >>> sets = pl.LazyFrame({
        ...     'id': ['a', 'b'],
        ...     'leadVariantId': ['1_1_A_T', '1_9_A_T'],
        ...     'qualityControls': [[], []],
        ... })
        >>> variants = pl.LazyFrame({'credibleSetId': ['a', 'b'], 'variantId': ['1_1_A_T', '1_2_A_T']})
        >>> flag_lead_variant_missing(sets, variants).sort('id').collect()['qualityControls'].list.len().to_list()
        [0, 1]
    """
    leads = variants.select(
        pl.col('credibleSetId').alias('id'), pl.col('variantId').alias('leadVariantId'), pl.lit(True).alias('_lead')
    ).unique()
    return (
        frame
        .join(leads, on=['id', 'leadVariantId'], how='left')
        .pipe(flag_rows, pl.col('_lead').is_null(), CredibleSetFlag.LEAD_VARIANT_MISSING)
        .drop('_lead')
    )


def set_qc(
    source: dict[str, str | list[str]],
    destination: str,
    settings: dict[str, Any],
    config: Config,
) -> None:
    """Add quality flags to the credible sets of one source.

    Args:
        source: The source's CredibleSet path under `credible_set`, its CredibleSetVariant path
            under `credible_set_variant`, and the validated study index under `study`.
        destination: Path of the flagged credible sets.
        settings: The accepted fine-mapping methods under `allowed_fm_methods`.
        config: Pipeline configuration.
    """
    sources = SetQcSource.model_validate(source)
    options = SetQcSettings.model_validate(settings)
    flagged = (
        scan_dataset(sources.credible_set)
        .match_to_schema(credible_set_schema)
        .pipe(flag_invalid_cs_id)
        .pipe(flag_invalid_chromosome)
        .pipe(flag_lead_chromosome_mismatch)
        .pipe(flag_in_mhc)
        .pipe(flag_invalid_finemapping_method, options.allowed_fm_methods)
        .pipe(flag_lead_variant_missing, scan_dataset(sources.credible_set_variant))
        .pipe(flag_study, scan_dataset(sources.study))
    )
    logger.info('Writing flagged credible sets to {}', destination)
    sink_dataset(flagged, destination).collect()
