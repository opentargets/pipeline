"""Quality checks of credible sets; the second credible set stage, which only adds flags."""

from typing import Any

import polars as pl
from loguru import logger
from otter.config.model import Config
from pydantic import BaseModel, ConfigDict, Field

from pts.schemas.credible_set import credible_set_schema
from pts.transformers.credible_set.enums import Chromosome, FinemappingMethod, KnownGenomicRegion
from pts.transformers.credible_set.flags import CredibleSetFlag
from pts.transformers.credible_set.variant_qc import CREDIBLE_SET_ID_PATTERN
from pts.transformers.utils.dataset import scan_dataset, sink_dataset
from pts.transformers.utils.quality_flags import flag_rows


class SetQcSource(BaseModel):
    """The credible sets of one source."""

    model_config = ConfigDict(extra='forbid')

    credible_set: str


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


def flag_sets(frame: pl.LazyFrame, allowed_fm_methods: list[FinemappingMethod]) -> pl.LazyFrame:
    """Add every set QC flag that depends only on the credible set row."""
    return (
        frame
        .pipe(flag_invalid_cs_id)
        .pipe(flag_invalid_chromosome)
        .pipe(flag_lead_chromosome_mismatch)
        .pipe(flag_in_mhc)
        .pipe(flag_invalid_finemapping_method, allowed_fm_methods)
    )


def set_qc(
    source: dict[str, str | list[str]],
    destination: str,
    settings: dict[str, Any],
    config: Config,
) -> None:
    """Add quality flags to the credible sets of one source.

    Args:
        source: The source's CredibleSet path under `credible_set`.
        destination: Path of the flagged credible sets.
        settings: The accepted fine-mapping methods under `allowed_fm_methods`.
        config: Pipeline configuration.
    """
    sources = SetQcSource.model_validate(source)
    options = SetQcSettings.model_validate(settings)
    flagged = (
        scan_dataset(sources.credible_set)
        .match_to_schema(credible_set_schema)
        .pipe(flag_sets, options.allowed_fm_methods)
    )
    logger.info('Writing flagged credible sets to {}', destination)
    sink_dataset(flagged, destination).collect()
