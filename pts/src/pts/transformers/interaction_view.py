"""Interaction view dataset generation.

One row per `targetA` in the served `interaction_view` shape: its interactions in `interaction` order,
each with its evidence in `interaction_evidence` order.
"""

from __future__ import annotations

from typing import Any

import polars as pl
from loguru import logger
from ot_datamodels.views.interaction_view import InteractionView, InteractionViewEvidence, InteractionViewRow
from otter.config.model import Config

from pts.schemas.dataset.conform import conform, without_enums
from pts.task_outputs import writes
from pts.transformers.utils.dataset import scan_dataset, write_dataset


def _unsigned(taxon_id: pl.Expr) -> pl.Expr:
    """`taxon_id mod 2^32`, as the served UInt32 holds IntAct's negative pseudo-taxids."""
    return taxon_id % 2**32


def _evidences(evidence: pl.LazyFrame) -> pl.LazyFrame:
    """`interactionId` and its served `evidences`, in evidence order."""
    fields = [
        _unsigned(pl.col(name)) if name == 'hostOrganismTaxId' else pl.col(name)
        for name in InteractionViewEvidence.model_fields
    ]
    return (
        evidence
        .select('interactionId', evidence=pl.struct(fields))
        .group_by('interactionId', maintain_order=True)
        .agg(evidences=pl.col('evidence'))
    )


def interaction_view_frame(interaction: pl.LazyFrame, evidence: pl.LazyFrame) -> pl.DataFrame:
    """Every `targetA` of `interaction` in `InteractionView` columns, before `conform`.

    Evidence naming no interaction is dropped, as the served LEFT JOIN from interactions did.

    Raises:
        ValueError: if an interaction has no evidence.
    """
    joined = (
        without_enums(interaction)
        .join(_evidences(without_enums(evidence)), on='interactionId', how='left', maintain_order='left')
        .collect()
    )
    missing = joined['evidences'].null_count()
    if missing:
        raise ValueError(f'interactions with no evidence: {missing}')
    species_b = pl.col('speciesB')
    row = [
        species_b.struct.with_fields(_unsigned(species_b.struct.field('taxonId')))
        if name == 'speciesB'
        else pl.col(name)
        for name in InteractionViewRow.model_fields
    ]
    return (
        joined
        .rechunk()
        .select('targetA', interaction=pl.struct(row))
        .group_by('targetA', maintain_order=True)
        .agg(interactions=pl.col('interaction'))
    )


@writes(view=InteractionView)
def interaction_view(
    source: dict[str, str],
    destination: dict[str, str],
    settings: dict[str, Any] | None,
    config: Config | None,
) -> None:
    """Build the interaction view and write it.

    Args:
        source: `interaction` and `evidence` locations.
        destination: `view`, the interaction view directory.
        settings: unused.
        config: unused.
    """
    logger.info('build interaction view')
    result = interaction_view_frame(scan_dataset(source['interaction']), scan_dataset(source['evidence']))
    logger.info(f'{result.height} targets in the interaction view')
    write_dataset(conform(result, InteractionView), str(destination['view']))
