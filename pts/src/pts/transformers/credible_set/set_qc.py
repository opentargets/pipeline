"""Set QC: the second credible set stage, which flags each credible set of one source.

Nothing leaves here; validation decides what the flags exclude. `set_qc` reads the source's sets
under their contract and runs the checks in order:

- the row itself: a credible set id that is not an md5, a chromosome that is not canonical, a lead
  on another chromosome than its set, a lead in the MHC, a fine-mapping method not accepted;
- against the validated study index: a study the index does not hold, a study type that is not
  one of the vocabulary or not the study's, and a curated top hit whose study has summary
  statistics; a set without a study type takes its study's.
"""

from __future__ import annotations

from typing import Any

import polars as pl
from loguru import logger
from otter.config.model import Config
from pydantic import BaseModel, ConfigDict, Field

from pts.schemas.credible_set import credible_set_required, credible_set_schema
from pts.schemas.study import StudyType
from pts.transformers.credible_set.flags import CredibleSetFlag
from pts.transformers.credible_set.identifier import CREDIBLE_SET_ID_PATTERN, variant_chromosome, variant_position
from pts.transformers.credible_set.source import read_source_dataset
from pts.transformers.credible_set.vocabulary import Chromosome, FinemappingMethod, KnownGenomicRegion
from pts.transformers.study import QualityFlag as StudyQualityFlag
from pts.transformers.utils.dataset import scan_dataset, sink_dataset
from pts.transformers.utils.quality_flags import flag_rows


class SetQcSource(BaseModel):
    """The credible sets of one source, and the validated study index."""

    model_config = ConfigDict(extra='forbid')

    credible_set: str
    study: str


class SetQcSettings(BaseModel):
    """Settings of set QC."""

    model_config = ConfigDict(extra='forbid')

    allowed_fm_methods: list[FinemappingMethod] = Field(default_factory=lambda: list(FinemappingMethod))
    """The accepted `fineMappingMethod` values; any other value gets INVALID_FINEMAPPING_METHOD."""


def set_qc(
    source: dict[str, str | list[str]],
    destination: str,
    settings: dict[str, Any],
    config: Config,
) -> None:
    """Add quality flags to the credible sets of one source.

    Args:
        source: The source's CredibleSet path under `credible_set`, and the validated study index
            under `study`.
        destination: Path of the flagged credible sets.
        settings: The accepted fine-mapping methods under `allowed_fm_methods`.
        config: Pipeline configuration.
    """
    sources = SetQcSource.model_validate(source)
    options = SetQcSettings.model_validate(settings)
    flagged = (
        read_source_dataset(sources.credible_set, credible_set_schema, credible_set_required)
        .pipe(_flag_invalid_cs_id)
        .pipe(_flag_invalid_chromosome)
        .pipe(_flag_lead_chromosome_mismatch)
        .pipe(_flag_in_mhc)
        .pipe(_flag_invalid_finemapping_method, options.allowed_fm_methods)
        .pipe(_flag_study, scan_dataset(sources.study))
    )
    logger.info('Writing flagged credible sets to {}', destination)
    sink_dataset(flagged, destination).collect()


def _flag_invalid_cs_id(frame: pl.LazyFrame) -> pl.LazyFrame:
    """Flag credible set identifiers that are not a 32-character lowercase hexadecimal md5.

    Examples:
        >>> credible_set = pl.LazyFrame({'id': ['0' * 32, 'A' * 32, '0' * 31], 'qualityControls': [[]] * 3})
        >>> _flag_invalid_cs_id(credible_set).collect()['qualityControls'].list.len().to_list()
        [0, 1, 1]
    """
    invalid = ~pl.col('id').str.contains(CREDIBLE_SET_ID_PATTERN)
    return flag_rows(frame, invalid, CredibleSetFlag.INVALID_CS_ID)


def _flag_invalid_chromosome(frame: pl.LazyFrame) -> pl.LazyFrame:
    """Flag credible sets whose `chromosome` is not one of `Chromosome` (1 to 22, X, Y, MT).

    Examples:
        >>> credible_set = pl.LazyFrame({'chromosome': ['1', 'XY', '23', 'chr1'], 'qualityControls': [[]] * 4})
        >>> _flag_invalid_chromosome(credible_set).collect()['qualityControls'].list.len().to_list()
        [0, 1, 1, 1]
    """
    invalid = ~pl.col('chromosome').is_in([chromosome.value for chromosome in Chromosome])
    return flag_rows(frame, invalid, CredibleSetFlag.INVALID_CHROMOSOME)


def _flag_lead_chromosome_mismatch(frame: pl.LazyFrame) -> pl.LazyFrame:
    """Flag credible sets whose lead variant lies on another chromosome than the set.

    The chromosome of the lead is the first field of `leadVariantId`.

    Examples:
        >>> credible_set = pl.LazyFrame({
        ...     'chromosome': ['1', '1', 'X'],
        ...     'leadVariantId': ['2_100_A_T', '1_100_A_T', '23_100_A_T'],
        ...     'qualityControls': [[]] * 3,
        ... })
        >>> _flag_lead_chromosome_mismatch(credible_set).collect()['qualityControls'].list.len().to_list()
        [1, 0, 1]
    """
    mismatch = variant_chromosome(pl.col('leadVariantId')) != pl.col('chromosome')
    return flag_rows(frame, mismatch, CredibleSetFlag.LEAD_CHROMOSOME_MISMATCH)


def _flag_in_mhc(frame: pl.LazyFrame) -> pl.LazyFrame:
    """Flag credible sets whose lead variant lies in the MHC region.

    The lead position is the second field of `leadVariantId`, and the region bounds of
    `KnownGenomicRegion.MHC` are inclusive.

    Examples:
        >>> credible_set = pl.LazyFrame({
        ...     'chromosome': ['6', '6', '6', '7'],
        ...     'leadVariantId': ['6_25726063_A_T', '6_33400556_A_T', '6_33400557_A_T', '7_30000000_A_T'],
        ...     'qualityControls': [[]] * 4,
        ... })
        >>> _flag_in_mhc(credible_set).collect()['qualityControls'].list.len().to_list()
        [1, 1, 0, 0]
    """
    mhc = KnownGenomicRegion.MHC.value
    lead_position = variant_position(pl.col('leadVariantId'))
    in_mhc = (pl.col('chromosome') == mhc.chromosome.value) & lead_position.is_between(mhc.start, mhc.end)
    return flag_rows(frame, in_mhc, CredibleSetFlag.IN_MHC)


def _flag_invalid_finemapping_method(frame: pl.LazyFrame, allowed: list[FinemappingMethod]) -> pl.LazyFrame:
    """Flag credible sets whose `fineMappingMethod` is not one of `allowed`; the comparison is exact.

    Examples:
        >>> credible_set = pl.LazyFrame({'fineMappingMethod': ['SuSiE', 'SuSie', 'PICS'], 'qualityControls': [[]] * 3})
        >>> flagged = _flag_invalid_finemapping_method(credible_set, [FinemappingMethod.SUSIE])
        >>> flagged.collect()['qualityControls'].list.len().to_list()
        [0, 1, 1]
    """
    invalid = ~pl.col('fineMappingMethod').is_in([method.value for method in allowed])
    return flag_rows(frame, invalid, CredibleSetFlag.INVALID_FINEMAPPING_METHOD)


def _flag_study(frame: pl.LazyFrame, study: pl.LazyFrame) -> pl.LazyFrame:
    """Flag credible sets against the study index, and fill a missing `studyType` from it.

    A set without a `studyType` takes its study's from the index, with no flag; a set that has one
    keeps it, so a mismatch stays visible.

    - INVALID_STUDY_TYPE: the source `studyType` is not one of `StudyType`; a missing type is not.
    - STUDY_TYPE_MISMATCH: the study is in the index, and the source `studyType` differs from the
      study's.
    - TOP_HIT_AND_SUMMARY_STATS: the set is a curated top hit, and the study is not flagged
      SUMSTATS_NOT_AVAILABLE, so its summary statistics should give the set instead.
    - MISSING_STUDY: the study is not in the study index.

    Examples:
        >>> credible_set = pl.LazyFrame({
        ...     'studyId': ['s', 't', 'u', 'unknown'],
        ...     'studyType': ['eQTL', 'gwas', None, 'gwas'],
        ...     'qualityControls': [[], [], [], []],
        ... })
        >>> study = pl.LazyFrame({
        ...     'studyId': ['s', 't', 'u'],
        ...     'studyType': ['eqtl', 'gwas', 'pqtl'],
        ...     'qualityControls': [[], [], []],
        ... })
        >>> for row in _flag_study(credible_set, study).sort('studyId').collect().iter_rows(named=True):
        ...     flag_name = sorted(CredibleSetFlag(flag).name for flag in row['qualityControls'])
        ...     print(row['studyId'], row['studyType'], flag_name)
        s eQTL ['INVALID_STUDY_TYPE', 'STUDY_TYPE_MISMATCH']
        t gwas []
        u pqtl []
        unknown gwas ['MISSING_STUDY']
    """
    lookup = study.select(
        'studyId',
        pl.col('studyType').alias('_study_type'),
        pl.col('qualityControls').alias('_study_qc'),
        pl.lit(True).alias('_study_found'),
    )
    found = pl.col('_study_found').is_not_null()
    mismatch = found & (pl.col('studyType') != pl.col('_study_type'))
    top_hit = pl.col('qualityControls').list.contains(CredibleSetFlag.TOP_HIT.value)
    no_sumstats = pl.col('_study_qc').list.contains(StudyQualityFlag.SUMSTATS_NOT_AVAILABLE.value)
    return (
        frame
        .join(lookup, on='studyId', how='left')
        .pipe(flag_rows, ~pl.col('studyType').is_in([t.value for t in StudyType]), CredibleSetFlag.INVALID_STUDY_TYPE)
        .pipe(flag_rows, mismatch, CredibleSetFlag.STUDY_TYPE_MISMATCH)
        .pipe(flag_rows, top_hit & ~no_sumstats, CredibleSetFlag.TOP_HIT_AND_SUMMARY_STATS)
        .pipe(flag_rows, ~found, CredibleSetFlag.MISSING_STUDY)
        .with_columns(pl.coalesce('studyType', '_study_type').alias('studyType'))
        .drop('_study_type', '_study_qc', '_study_found')
    )
