"""Mechanisms of action by drug: the served `drug_mechanism_of_action_by_drug_view` shape.

One row per molecule in a mechanism's `chemblIds`: the mechanisms naming it by `mechanismOfAction, actionType,
targetName, curation` (Enums in vocabulary order, ties in output order), and their distinct target and action types in
vocabulary order. A mechanism naming no molecule is in no row; one naming a molecule twice is in its row twice.
"""

from __future__ import annotations

from typing import Any

import polars as pl
from loguru import logger
from ot_datamodels.views.drug_mechanism_of_action_by_drug_view import MECHANISM_FIELDS, DrugMechanismOfActionByDrugView
from otter.config.model import Config

from pts.schemas.dataset.conform import conform
from pts.task_outputs import writes
from pts.transformers.utils.dataset import scan_dataset, write_dataset


def drug_mechanism_of_action_by_drug_frame(mechanisms: pl.DataFrame) -> pl.DataFrame:
    """Every molecule of `drug_mechanism_of_action` with its mechanisms and their types; rows by molecule."""
    return (
        mechanisms
        .rechunk()
        .filter(pl.col('chemblIds').list.len() > 0)
        .explode('chemblIds')
        .rename({'chemblIds': 'chemblId'})
        .sort('chemblId', 'mechanismOfAction', 'actionType', 'targetName', 'curation', maintain_order=True)
        .select('chemblId', 'targetType', 'actionType', item=pl.struct(MECHANISM_FIELDS))
        .group_by('chemblId', maintain_order=True)
        .agg(
            rows=pl.col('item'),
            uniqueTargetTypes=pl.col('targetType').unique().sort(),
            uniqueActionTypes=pl.col('actionType').unique().sort(),
        )
    )


@writes(view=DrugMechanismOfActionByDrugView)
def drug_mechanism_of_action_by_drug(
    source: str,
    destination: dict[str, str],
    settings: dict[str, Any] | None,
    config: Config | None,
) -> None:
    """Build the mechanisms of action by drug view and write it.

    Args:
        source: the `drug_mechanism_of_action` output.
        destination: `view`, the view directory.
        settings: unused.
        config: unused.
    """
    mechanisms = scan_dataset(str(source)).collect()
    dropped = mechanisms.filter(pl.col('chemblIds').list.len() == 0).height
    result = drug_mechanism_of_action_by_drug_frame(mechanisms)
    logger.info(f'{result.height} molecules in the mechanism of action view; {dropped} mechanisms name no molecule')
    write_dataset(conform(result, DrugMechanismOfActionByDrugView), str(destination['view']))
