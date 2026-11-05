"""Annotate the validated credible sets and variants, and publish them."""

from typing import Any

import polars as pl
from loguru import logger
from otter.config.model import Config
from pydantic import BaseModel, ConfigDict

from pts.transformers.credible_set.enums import FinemappingMethod
from pts.transformers.credible_set.flags import CredibleSetConfidenceClasses, CredibleSetFlag
from pts.transformers.utils.dataset import scan_dataset, sink_dataset


class AnnotationSource(BaseModel):
    """The validated sets and variants of every source."""

    model_config = ConfigDict(extra='forbid')

    credible_set: str
    credible_set_variant: str


class AnnotationDestination(BaseModel):
    """Paths of the published credible sets and variants."""

    model_config = ConfigDict(extra='forbid')

    credible_set: str
    credible_set_variant: str


def assign_confidence(frame: pl.LazyFrame) -> pl.LazyFrame:
    """Set `confidence` from the fine-mapping method and the LD and top hit flags, as in gentropy.

    The conditions are tested in order:

    - SuSiE or SuSiE-inf without `OUT_OF_SAMPLE_LD`: 4, `FINEMAPPED_IN_SAMPLE_LD`;
    - SuSiE or SuSiE-inf with `OUT_OF_SAMPLE_LD`: 3, `FINEMAPPED_OUT_OF_SAMPLE_LD`;
    - PICS without `TOP_HIT`: 2, `PICSED_SUMMARY_STATS`;
    - PICS with `TOP_HIT`: 1, `PICSED_TOP_HIT`;
    - any other set: 0, `UNKNOWN`.

    Examples:
        >>> out_of_sample, top_hit = CredibleSetFlag.OUT_OF_SAMPLE_LD.value, CredibleSetFlag.TOP_HIT.value
        >>> sets = pl.LazyFrame({
        ...     'fineMappingMethod': ['SuSiE', 'SuSiE-inf', 'PICS', 'PICS', 'SuSie'],
        ...     'qualityControls': [[], [out_of_sample], [], [top_hit], []],
        ... })
        >>> assign_confidence(sets).collect()['confidence'].to_list()
        [4, 3, 2, 1, 0]
    """
    flags = pl.col('qualityControls')
    susie = pl.col('fineMappingMethod').is_in([method.value for method in FinemappingMethod.susie()])
    pics = pl.col('fineMappingMethod') == FinemappingMethod.PICS.value
    classes = CredibleSetConfidenceClasses
    confidence = (
        pl
        .when(susie & ~flags.list.contains(CredibleSetFlag.OUT_OF_SAMPLE_LD.value))
        .then(classes.FINEMAPPED_IN_SAMPLE_LD.value)
        .when(susie)
        .then(classes.FINEMAPPED_OUT_OF_SAMPLE_LD.value)
        .when(pics & ~flags.list.contains(CredibleSetFlag.TOP_HIT.value))
        .then(classes.PICSED_SUMMARY_STATS.value)
        .when(pics)
        .then(classes.PICSED_TOP_HIT.value)
        .otherwise(classes.UNKNOWN.value)
    )
    return frame.with_columns(confidence.cast(pl.Int8).alias('confidence'))


def annotation(
    source: dict[str, str | list[str]],
    destination: dict[str, str],
    settings: dict[str, Any],
    config: Config,
) -> None:
    """Annotate the validated credible sets and write the published datasets.

    Args:
        source: The validated sets under `credible_set` and the validated variants under
            `credible_set_variant`.
        destination: Published paths under `credible_set` and `credible_set_variant`.
        settings: Unused.
        config: Pipeline configuration.
    """
    sources = AnnotationSource.model_validate(source)
    destinations = AnnotationDestination.model_validate(destination)
    sets = scan_dataset(sources.credible_set).pipe(assign_confidence)
    variants = scan_dataset(sources.credible_set_variant)
    logger.info(
        'Writing credible sets to {} and variants to {}', destinations.credible_set, destinations.credible_set_variant
    )
    pl.collect_all([
        sink_dataset(sets, destinations.credible_set),
        sink_dataset(variants, destinations.credible_set_variant),
    ])
