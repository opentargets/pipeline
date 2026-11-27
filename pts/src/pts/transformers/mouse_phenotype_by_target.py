"""Mouse phenotypes by target: the served `mouse_phenotype_by_target_view` shape.

One row per human target, its `mouse_phenotype` rows nested by `modelPhenotypeId, targetInModelMgiId`.
"""

from __future__ import annotations

from typing import Any

import polars as pl
from loguru import logger
from ot_datamodels.views.mouse_phenotype_by_target_view import MousePhenotypeByTargetView, MousePhenotypeItem
from otter.config.model import Config

from pts.schemas.dataset.conform import conform
from pts.task_outputs import writes
from pts.transformers.utils.dataset import scan_dataset, write_dataset


def mouse_phenotype_by_target_frame(phenotypes: pl.DataFrame) -> pl.DataFrame:
    """Every target of `mouse_phenotype` with its rows, by `modelPhenotypeId, targetInModelMgiId`; rows by target."""
    return (
        phenotypes
        .rechunk()
        .sort('targetFromSourceId', 'modelPhenotypeId', 'targetInModelMgiId')
        .select('targetFromSourceId', item=pl.struct(list(MousePhenotypeItem.model_fields)))
        .group_by('targetFromSourceId', maintain_order=True)
        .agg(mouse_phenotypes=pl.col('item'))
    )


@writes(view=MousePhenotypeByTargetView)
def mouse_phenotype_by_target(
    source: str,
    destination: dict[str, str],
    settings: dict[str, Any] | None,
    config: Config | None,
) -> None:
    """Build the mouse phenotypes by target view and write it.

    Args:
        source: the `mouse_phenotype` output.
        destination: `view`, the view directory.
        settings: unused.
        config: unused.
    """
    result = mouse_phenotype_by_target_frame(scan_dataset(str(source)).collect())
    logger.info(f'{result.height} targets in the mouse phenotype view')
    write_dataset(conform(result, MousePhenotypeByTargetView), str(destination['view']))
