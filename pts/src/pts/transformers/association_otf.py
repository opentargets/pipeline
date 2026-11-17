"""Association on-the-fly (OTF) rows: every evidence row with its disease and target labels and novelties."""

from __future__ import annotations

from typing import Any

import polars as pl
from loguru import logger

from pts.schemas.association import association_otf_schema
from pts.transformers.utils.dataset import scan_dataset, scan_datasets, sink_dataset

EVIDENCE_COLUMNS = ['id', 'diseaseId', 'targetId', 'datasourceId', 'datatypeId', 'score']


def association_otf(source: dict[str, str], destination: str, settings: Any, config: Any) -> None:
    """Enrich evidence with disease and target labels and the overall novelties.

    Args:
        source: paths keyed by evidence (a glob), disease, target, association_direct and association_indirect.
        destination: output dataset directory.
        settings: unused.
        config: unused.
    """
    evidence = scan_datasets(source['evidence']).select(EVIDENCE_COLUMNS).rename({'id': 'rowId', 'score': 'rowScore'})
    disease = scan_dataset(source['disease']).select(
        pl.col('id').alias('diseaseId'),
        pl.concat_str('id', pl.lit(' '), 'name').alias('diseaseData'),
    )
    target = scan_dataset(source['target']).select(
        pl.col('id').alias('targetId'),
        pl.concat_str('id', pl.lit(' '), 'approvedName', pl.lit(' '), 'approvedSymbol').alias('targetData'),
    )

    def novelty(path: str, name: str) -> pl.LazyFrame:
        return scan_dataset(path).select('targetId', 'diseaseId', pl.col('currentNovelty').alias(name))

    result = (
        evidence
        .join(disease, on='diseaseId', how='left')
        .join(target, on='targetId', how='left')
        .join(novelty(source['association_direct'], 'noveltyDirect'), on=['diseaseId', 'targetId'], how='left')
        .join(novelty(source['association_indirect'], 'noveltyIndirect'), on=['diseaseId', 'targetId'], how='left')
        .select(list(association_otf_schema()))
        .cast(association_otf_schema())
        .sort('rowId')
    )
    logger.info('Writing association OTF data to {}', destination)
    sink_dataset(result, destination).collect()
