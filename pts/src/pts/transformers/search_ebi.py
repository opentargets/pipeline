"""Search EBI dataset generation.

Ported from the platform-etl-backend SearchEBI step. Joins associations and GWAS credible set evidence
with target and disease metadata for the EBI search index.
"""

from __future__ import annotations

from typing import Any

import polars as pl
from loguru import logger
from otter.config.model import Config

from pts.transformers.utils.dataset import scan_dataset, sink_dataset

SORT_KEYS = ['targetId', 'diseaseId', 'score']


def generate_datasets(
    diseases: pl.LazyFrame,
    targets: pl.LazyFrame,
    associations: pl.LazyFrame,
    evidence: pl.LazyFrame,
) -> dict[str, pl.LazyFrame]:
    """Join associations and evidence with target and disease metadata.

    Args:
        diseases: diseases keyed by `diseaseId`.
        targets: targets keyed by `targetId`.
        associations: associations with `targetId`, `diseaseId` and `associationScore`.
        evidence: evidence with `targetId`, `diseaseId` and `score`.

    Returns:
        The `associations` and `evidence` datasets, keeping only rows whose target and disease are known.
    """
    targets = targets.select('targetId', 'approvedSymbol')
    diseases = diseases.select('diseaseId', 'name')
    assoc = associations.select('targetId', 'diseaseId', pl.col('associationScore').alias('score'))
    evid = evidence.select('targetId', 'diseaseId', 'score')

    def join(lf: pl.LazyFrame) -> pl.LazyFrame:
        return (
            lf
            .join(targets, 'targetId', 'inner')
            .join(diseases, 'diseaseId', 'inner')
            .select('targetId', 'diseaseId', 'approvedSymbol', 'name', 'score')
            .sort(SORT_KEYS)
        )

    return {'associations': join(assoc), 'evidence': join(evid)}


def search_ebi(
    source: dict[str, str],
    destination: dict[str, str],
    settings: dict[str, Any] | None,
    config: Config | None,
) -> None:
    """Generate the search EBI association and evidence views.

    Args:
        source: paths keyed by disease, target, association and evidence.
        destination: paths keyed by associations and evidence.
        settings: unused.
        config: unused.
    """
    diseases = scan_dataset(source['disease']).rename({'id': 'diseaseId'})
    targets = scan_dataset(source['target']).rename({'id': 'targetId'})
    associations = scan_dataset(source['association'])
    evidence = scan_dataset(source['evidence'])

    datasets = generate_datasets(diseases, targets, associations, evidence)

    logger.info(f'writing {destination["associations"]} and {destination["evidence"]}')
    pl.collect_all([sink_dataset(datasets[name], destination[name]) for name in ('associations', 'evidence')])
