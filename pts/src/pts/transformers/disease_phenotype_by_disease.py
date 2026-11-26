"""Disease phenotypes by disease: the served `disease_hpo` shape.

One row per disease, its `disease_phenotype` rows nested by `phenotype`.
"""

from __future__ import annotations

from typing import Any

import polars as pl
from loguru import logger
from ot_datamodels.views.disease_phenotype_by_disease_view import DiseasePhenotypeByDiseaseView, DiseasePhenotypeItem
from otter.config.model import Config

from pts.schemas.dataset.conform import conform
from pts.task_outputs import writes
from pts.transformers.utils.dataset import scan_dataset, write_dataset


def disease_phenotype_by_disease_frame(phenotypes: pl.DataFrame) -> pl.DataFrame:
    """Every disease of `disease_phenotype` with its rows, by `phenotype`; rows by `disease`."""
    return (
        phenotypes
        .rechunk()
        .sort('disease', 'phenotype')
        .select('disease', item=pl.struct(list(DiseasePhenotypeItem.model_fields)))
        .group_by('disease', maintain_order=True)
        .agg(phenotypes=pl.col('item'))
    )


@writes(view=DiseasePhenotypeByDiseaseView)
def disease_phenotype_by_disease(
    source: str,
    destination: dict[str, str],
    settings: dict[str, Any] | None,
    config: Config | None,
) -> None:
    """Build the disease phenotypes by disease view and write it.

    Args:
        source: the `disease_phenotype` output.
        destination: `view`, the view directory.
        settings: unused.
        config: unused.
    """
    result = disease_phenotype_by_disease_frame(scan_dataset(str(source)).collect())
    logger.info(f'{result.height} diseases in the disease phenotype view')
    write_dataset(conform(result, DiseasePhenotypeByDiseaseView), str(destination['view']))
