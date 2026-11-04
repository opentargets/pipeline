"""Annotate the validated credible sets and variants, and publish them."""

from __future__ import annotations

from typing import Any

import polars as pl
from loguru import logger
from otter.config.model import Config
from pydantic import BaseModel, ConfigDict, Field

from pts.transformers.credible_set.allele_frequency import effect_allele_frequency
from pts.transformers.credible_set.confidence import assign_confidence
from pts.transformers.credible_set.flags import CredibleSetFlag
from pts.transformers.credible_set.replication import replicated_sets
from pts.transformers.credible_set.rescaling import rescaled_effect_size
from pts.transformers.credible_set.trans_qtl import trans_qtls
from pts.transformers.utils.dataset import scan_dataset, sink_dataset
from pts.transformers.utils.quality_flags import flag_matching_ids


class AnnotationSource(BaseModel):
    """The validated sets and variants of every source, the study index and the variant index."""

    model_config = ConfigDict(extra='forbid')

    credible_set: str
    credible_set_variant: str
    study: str
    """The study index, for the populations of each study."""
    variant: str
    """The variant index, for the gnomAD allele frequencies."""
    target: str
    """The target index, for the chromosome and TSS of the gene of a QTL study."""
    disease: str
    """The disease index, for the therapeutic areas that mark a term as a measurement."""


class AnnotationDestination(BaseModel):
    """Paths of the published credible sets and variants."""

    model_config = ConfigDict(extra='forbid')

    credible_set: str
    credible_set_variant: str


class AnnotationSettings(BaseModel):
    """Settings of the annotation."""

    model_config = ConfigDict(extra='forbid')

    trans_qtl_threshold: int = Field(default=5_000_000, ge=0)
    """The distance from the gene TSS beyond which a QTL variant is trans."""


def annotation(
    source: dict[str, str | list[str]],
    destination: dict[str, str],
    settings: dict[str, Any],
    config: Config,
) -> None:
    """Annotate the validated credible sets and write the published datasets.

    Args:
        source: The validated sets under `credible_set`, the validated variants under
            `credible_set_variant`, and the study, variant, target and disease indexes under `study`,
            `variant`, `target` and `disease`.
        destination: Published paths under `credible_set` and `credible_set_variant`.
        settings: The distance from the gene TSS beyond which a QTL variant is trans, under
            `trans_qtl_threshold`.
        config: Pipeline configuration.
    """
    sources = AnnotationSource.model_validate(source)
    destinations = AnnotationDestination.model_validate(destination)
    options = AnnotationSettings.model_validate(settings)
    validated_set = scan_dataset(sources.credible_set)
    validated_variant = scan_dataset(sources.credible_set_variant)
    study = scan_dataset(sources.study)
    target = scan_dataset(sources.target)
    disease = scan_dataset(sources.disease).select('id', 'therapeuticAreas')
    trans_set = trans_qtls(validated_set, validated_variant, study, target, options.trans_qtl_threshold)
    replicated = replicated_sets(validated_set, study, disease)
    credible_set = (
        validated_set
        .pipe(assign_confidence)
        .pipe(flag_matching_ids, 'id', trans_set, CredibleSetFlag.TRANS_QTL)
        .pipe(flag_matching_ids, 'id', replicated, CredibleSetFlag.REPLICATED)
    )
    credible_set_variant = validated_variant.pipe(
        effect_allele_frequency, validated_set, study, scan_dataset(sources.variant)
    ).pipe(rescaled_effect_size, validated_set, study)
    logger.info(
        'Writing credible sets to {} and variants to {}', destinations.credible_set, destinations.credible_set_variant
    )
    pl.collect_all([
        sink_dataset(credible_set, destinations.credible_set),
        sink_dataset(credible_set_variant, destinations.credible_set_variant),
    ])
