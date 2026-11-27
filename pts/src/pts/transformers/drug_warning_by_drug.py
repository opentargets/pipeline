"""Drug warnings by drug: the served `drug_warning_by_drug_view` shape.

One row per molecule in a warning's `chemblIds`, the warnings naming it nested by `id`. A warning naming no molecule
is in no row; one naming a molecule twice is in its row twice.
"""

from __future__ import annotations

from typing import Any

import polars as pl
from loguru import logger
from ot_datamodels.views.drug_warning_by_drug_view import DrugWarningByDrugView, DrugWarningItem
from otter.config.model import Config

from pts.schemas.dataset.conform import conform
from pts.task_outputs import writes
from pts.transformers.utils.dataset import scan_dataset, write_dataset


def drug_warning_by_drug_frame(warnings: pl.DataFrame) -> pl.DataFrame:
    """Every molecule of `drug_warning` with the warnings naming it, by `id`; rows by molecule."""
    return (
        warnings
        .rechunk()
        .filter(pl.col('chemblIds').list.len() > 0)
        .with_columns(chemblId=pl.col('chemblIds'))
        .explode('chemblId')
        .sort('chemblId', 'id', maintain_order=True)
        .select('chemblId', item=pl.struct(list(DrugWarningItem.model_fields)))
        .group_by('chemblId', maintain_order=True)
        .agg(drugWarnings=pl.col('item'))
    )


@writes(view=DrugWarningByDrugView)
def drug_warning_by_drug(
    source: str,
    destination: dict[str, str],
    settings: dict[str, Any] | None,
    config: Config | None,
) -> None:
    """Build the drug warnings by drug view and write it.

    Args:
        source: the `drug_warning` output.
        destination: `view`, the view directory.
        settings: unused.
        config: unused.
    """
    warnings = scan_dataset(str(source)).collect()
    dropped = warnings.filter(pl.col('chemblIds').list.len() == 0).height
    result = drug_warning_by_drug_frame(warnings)
    logger.info(f'{result.height} molecules in the drug warning view; {dropped} warnings name no molecule')
    write_dataset(conform(result, DrugWarningByDrugView), str(destination['view']))
