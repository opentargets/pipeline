"""Drug view dataset generation.

Every drug molecule row with the ids of the pharmacogenomics records that name it.
"""

from __future__ import annotations

from typing import Any

import polars as pl
from loguru import logger
from ot_datamodels.views.drug_view import DrugView
from otter.config.model import Config

from pts.schemas.dataset.conform import conform, without_enums
from pts.task_outputs import writes
from pts.transformers.utils.dataset import scan_dataset, write_dataset


def drug_view_frame(drug_molecule: pl.DataFrame, pharmacogenomics: pl.DataFrame) -> pl.DataFrame:
    """Every drug_molecule row, in its order, with `pharmacogenomicsIds` last.

    The ids of the pharmacogenomics rows naming the drug, ascending, once per mention; `[]` for none.
    Null drug ids and drugs absent from drug_molecule are ignored.
    """
    ids = (
        pharmacogenomics
        .select('pharmacogenomicsId', 'drugs')
        .explode('drugs')
        .select('pharmacogenomicsId', id=pl.col('drugs').struct.field('drugId'))
        .drop_nulls('id')
        .group_by('id')
        .agg(pharmacogenomicsIds=pl.col('pharmacogenomicsId').sort())
    )
    empty = pl.lit([], dtype=pl.List(pharmacogenomics.schema['pharmacogenomicsId']))
    return drug_molecule.join(ids, on='id', how='left', maintain_order='left').with_columns(
        pl.col('pharmacogenomicsIds').fill_null(empty)
    )


@writes(view=DrugView)
def drug_view(
    source: dict[str, str],
    destination: dict[str, str],
    settings: dict[str, Any] | None,
    config: Config | None,
) -> None:
    """Build the drug view and write it.

    Args:
        source: `drug_molecule` and `pharmacogenomics` locations.
        destination: `view`, the drug view directory.
        settings: unused.
        config: unused.
    """
    logger.info('load drug view inputs')
    drug_molecule, pharmacogenomics = pl.collect_all([
        scan_dataset(source['drug_molecule']),
        scan_dataset(source['pharmacogenomics']).select('pharmacogenomicsId', 'drugs'),
    ])
    result = drug_view_frame(without_enums(drug_molecule), pharmacogenomics)
    logger.info(f'{result.height} drugs in the drug view')
    write_dataset(conform(result, DrugView), str(destination['view']))
