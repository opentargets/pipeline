"""Association on-the-fly (OTF) rows: every evidence row keyed both ways, with search labels and novelties."""

from __future__ import annotations

import string
from typing import Any

import polars as pl
from loguru import logger
from ot_datamodels.views.association_otf_disease import AssociationOtfDisease
from ot_datamodels.views.association_otf_target import AssociationOtfTarget

from pts.schemas.dataset.conform import conform
from pts.task_outputs import writes
from pts.transformers.utils.dataset import scan_dataset, scan_datasets, sink_dataset

EVIDENCE_COLUMNS = ['id', 'diseaseId', 'targetId', 'datasourceId', 'datatypeId', 'score']
MEASUREMENT = 'EFO_0001444'
_EMPTY = pl.lit([], dtype=pl.List(pl.String))


def ascii_lower(expr: pl.Expr) -> pl.Expr:
    """`expr` with `A`-`Z` lower-cased and every other character kept, as ClickHouse `lower()` does."""
    return expr.str.replace_many(list(string.ascii_uppercase), list(string.ascii_lowercase))


def _evidence(source: dict[str, str]) -> pl.LazyFrame:
    """Every evidence row with its labels, novelties, disease ancestors and measurement flag, by `rowId`."""
    evidence = scan_datasets(source['evidence']).select(EVIDENCE_COLUMNS).rename({'id': 'rowId', 'score': 'rowScore'})
    disease = scan_dataset(source['disease']).select(
        pl.col('id').alias('diseaseId'),
        ascii_lower(pl.concat_str('id', pl.lit(' '), 'name')).alias('diseaseData'),
        'ancestors',
        pl.col('therapeuticAreas').list.contains(MEASUREMENT).alias('isMeasurement'),
    )
    target = scan_dataset(source['target']).select(
        pl.col('id').alias('targetId'),
        ascii_lower(pl.concat_str('id', pl.lit(' '), 'approvedName', pl.lit(' '), 'approvedSymbol')).alias(
            'targetData'
        ),
    )

    def novelty(path: str, name: str) -> pl.LazyFrame:
        return scan_dataset(path).select('targetId', 'diseaseId', pl.col('currentNovelty').alias(name))

    return (
        evidence
        .join(disease, on='diseaseId', how='left')
        .join(target, on='targetId', how='left')
        .join(novelty(source['association_direct'], 'noveltyDirect'), on=['diseaseId', 'targetId'], how='left')
        .join(novelty(source['association_indirect'], 'noveltyIndirect'), on=['diseaseId', 'targetId'], how='left')
        .sort('rowId')
    )


def _side(evidence: pl.LazyFrame, a: str, b: str, search_a: str, search_b: str, *rest: str | pl.Expr) -> pl.LazyFrame:
    return evidence.select(
        'rowId',
        pl.col(a).alias('A'),
        pl.col(b).alias('B'),
        'datatypeId',
        'datasourceId',
        'rowScore',
        pl.col(search_a).alias('searchA'),
        pl.col(search_b).alias('searchB'),
        *rest,
    )


@writes(disease=AssociationOtfDisease, target=AssociationOtfTarget)
def association_otf(source: dict[str, str], destination: dict[str, str], settings: Any, config: Any) -> None:
    """Write every evidence row keyed by disease (`disease`) and by target (`target`), as served.

    Args:
        source: paths keyed by evidence (a glob), disease, target, association_direct and association_indirect.
        destination: `disease` and `target` dataset directories.
        settings: unused.
        config: unused.
    """
    evidence = _evidence(source)
    novelties = ('noveltyDirect', 'noveltyIndirect')
    disease_side = _side(
        evidence, 'diseaseId', 'targetId', 'diseaseData', 'targetData', *novelties, _EMPTY.alias('indirect')
    )
    target_side = _side(
        evidence,
        'targetId',
        'diseaseId',
        'targetData',
        'diseaseData',
        pl.col('isMeasurement').fill_null(False),
        *novelties,
        pl.col('ancestors').fill_null(_EMPTY).alias('indirect'),
    )
    logger.info('Writing association OTF data to {} and {}', destination['disease'], destination['target'])
    pl.collect_all([
        sink_dataset(conform(disease_side, AssociationOtfDisease), destination['disease']),
        sink_dataset(conform(target_side, AssociationOtfTarget), destination['target']),
    ])
