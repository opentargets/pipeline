"""Quality checks of credible sets; the second credible set stage, which only adds flags."""

from typing import Any

import polars as pl
from loguru import logger
from otter.config.model import Config
from pydantic import BaseModel, ConfigDict

from pts.schemas.credible_set import credible_set_schema
from pts.transformers.credible_set.enums import KnownGenomicRegion
from pts.transformers.credible_set.flags import CredibleSetFlag
from pts.transformers.utils.dataset import scan_dataset, sink_dataset
from pts.transformers.utils.quality_flags import flag_rows


class SetQcSource(BaseModel):
    """The credible sets of one source."""

    model_config = ConfigDict(extra='forbid')

    credible_set: str


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
        settings: Unused.
        config: Pipeline configuration.
    """
    sources = SetQcSource.model_validate(source)
    flagged = scan_dataset(sources.credible_set).match_to_schema(credible_set_schema).pipe(flag_in_mhc)
    logger.info('Writing flagged credible sets to {}', destination)
    sink_dataset(flagged, destination).collect()
