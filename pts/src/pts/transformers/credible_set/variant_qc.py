"""Quality checks of credible set variants; the first credible set stage, which only adds flags."""

from typing import Any

import polars as pl
from loguru import logger
from otter.config.model import Config
from pydantic import BaseModel, ConfigDict

from pts.schemas.credible_set import credible_set_variant_schema
from pts.transformers.credible_set.enums import Chromosome
from pts.transformers.credible_set.flags import CredibleSetVariantFlag
from pts.transformers.utils.dataset import scan_dataset, sink_dataset
from pts.transformers.utils.quality_flags import flag_rows

#: A variant identifier on a canonical chromosome, with a positive position and nucleotide alleles.
VARIANT_ID_PATTERN = rf'^(?:{"|".join(chromosome.value for chromosome in Chromosome)})_[1-9][0-9]*_[ACGT]+_[ACGT]+$'


class VariantQcSource(BaseModel):
    """The credible set variants of one source."""

    model_config = ConfigDict(extra='forbid')

    credible_set_variant: str


def flag_invalid_variant_id(frame: pl.LazyFrame) -> pl.LazyFrame:
    """Flag variant identifiers that are not `chromosome_position_ref_alt`, or whose ref equals alt.

    A valid identifier has a chromosome of `Chromosome`, a position from 1, and reference and
    alternative alleles of the nucleotides A, C, G and T that differ.

    Examples:
        >>> variants = pl.LazyFrame({
        ...     'variantId': [
        ...         '1_100_A_T', '22_100_A_T', 'XY_5_ACGT_A', '1_100_A_A', '23_100_A_T', '14_KI270846v1_alt_497149_C_T'
        ...     ],
        ...     'qualityControls': [[]] * 6,
        ... })
        >>> flagged = flag_invalid_variant_id(variants).collect()
        >>> flagged.filter(pl.col('qualityControls').list.len() > 0)['variantId'].to_list()
        ['1_100_A_A', '23_100_A_T', '14_KI270846v1_alt_497149_C_T']
    """
    alleles = pl.col('variantId').str.split('_')
    invalid = ~pl.col('variantId').str.contains(VARIANT_ID_PATTERN) | (alleles.list.get(2) == alleles.list.get(3))
    return flag_rows(frame, invalid, CredibleSetVariantFlag.INVALID_VARIANT_ID)


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
        settings: Unused.
        config: Pipeline configuration.
    """
    sources = VariantQcSource.model_validate(source)
    flagged = (
        scan_dataset(sources.credible_set_variant)
        .match_to_schema(credible_set_variant_schema)
        .pipe(flag_invalid_variant_id)
    )
    logger.info('Writing flagged credible set variants to {}', destination)
    sink_dataset(flagged, destination).collect()
