"""Search EBI dataset generation.

Ported from the platform-etl-backend SearchEBI step. Joins associations with target and disease
metadata and exports them as a JSON-lines file for the EBI search index.
"""

from __future__ import annotations

from typing import Any

import polars as pl
from loguru import logger
from otter.config.model import Config

from pts.schemas.dataset.conform import without_enums
from pts.transformers.utils.dataset import scan_dataset, write_json_file

SORT_KEYS = ['targetId', 'diseaseId', 'score']


def generate_associations(
    diseases: pl.LazyFrame,
    targets: pl.LazyFrame,
    associations: pl.LazyFrame,
) -> pl.LazyFrame:
    """Join associations with target and disease metadata.

    Args:
        diseases: diseases keyed by `diseaseId`.
        targets: targets keyed by `targetId`.
        associations: associations with `targetId`, `diseaseId` and `associationScore`.

    Returns:
        Associations whose target and disease are known, sorted by `SORT_KEYS`.
    """
    return (
        associations
        .select('targetId', 'diseaseId', pl.col('associationScore').alias('score'))
        .join(targets.select('targetId', 'approvedSymbol'), 'targetId', 'inner')
        .join(diseases.select('diseaseId', 'name'), 'diseaseId', 'inner')
        .select('targetId', 'diseaseId', 'approvedSymbol', 'name', 'score')
        .sort(SORT_KEYS)
    )


def search_ebi(
    source: dict[str, str],
    destination: dict[str, str],
    settings: dict[str, Any] | None,
    config: Config | None,
) -> None:
    """Generate the search EBI associations export, one JSON-lines file.

    Args:
        source: paths keyed by disease, target and association.
        destination: path keyed by associations.
        settings: unused.
        config: unused.
    """
    diseases = scan_dataset(source['disease']).rename({'id': 'diseaseId'})
    targets = without_enums(scan_dataset(source['target'])).rename({'id': 'targetId'})
    associations = without_enums(scan_dataset(source['association']))

    logger.info(f'writing {destination["associations"]}')
    write_json_file(generate_associations(diseases, targets, associations), destination['associations'])
