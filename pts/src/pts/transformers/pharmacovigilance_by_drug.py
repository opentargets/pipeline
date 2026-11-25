"""Pharmacovigilance by drug: the served `pharmacovigilance` shape.

One row per drug: its adverse events by `logLR` descending, then `name`, and `criticalValue`, the first event's
`critval`.
"""

from __future__ import annotations

from typing import Any

import polars as pl
from loguru import logger
from ot_datamodels.views.pharmacovigilance_by_drug import PharmacovigilanceByDrug
from otter.config.model import Config

from pts.schemas.dataset.conform import conform
from pts.task_outputs import writes
from pts.transformers.utils.dataset import scan_dataset, write_dataset


def pharmacovigilance_by_drug_frame(events: pl.DataFrame) -> pl.DataFrame:
    """Every drug of `pharmacovigilance` with its adverse events; rows by drug."""
    event = pl.struct(
        name=pl.col('event'),
        count=pl.col('count'),
        logLR=pl.col('llr'),
        critval=pl.col('critval'),
        meddraCode=pl.col('meddraCode'),
    )
    return (
        events
        .rechunk()
        .sort(['chembl_id', 'llr', 'event'], descending=[False, True, False])
        .select('chembl_id', 'critval', event=event)
        .group_by('chembl_id', maintain_order=True)
        .agg(adverse_events=pl.col('event'), criticalValue=pl.col('critval').first())
    )


@writes(view=PharmacovigilanceByDrug)
def pharmacovigilance_by_drug(
    source: str,
    destination: dict[str, str],
    settings: dict[str, Any] | None,
    config: Config | None,
) -> None:
    """Build the pharmacovigilance by drug view and write it.

    Args:
        source: the `pharmacovigilance` output.
        destination: `view`, the view directory.
        settings: unused.
        config: unused.
    """
    result = pharmacovigilance_by_drug_frame(scan_dataset(str(source)).collect())
    logger.info(f'{result.height} drugs in the pharmacovigilance view')
    write_dataset(conform(result, PharmacovigilanceByDrug), str(destination['view']))
