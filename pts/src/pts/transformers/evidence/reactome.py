"""Reactome evidence."""

from __future__ import annotations

from typing import Any

import polars as pl
from otter.config.model import Config

from pts.transformers.evidence.utils.datasource import EvidenceDatasource
from pts.transformers.evidence.utils.run import run_evidence
from pts.transformers.utils.dataset import scan_dataset

REACTOME = EvidenceDatasource(
    datasource_id='reactome',
    # order feeds the id, as in the spark config
    unique_fields=[
        'targetId',
        'targetFromSourceId',
        'diseaseId',
        'datasourceId',
        'variantAminoacidDescriptions',
        'targetModulation',
        'reactionId',
    ],
    score=pl.lit(1.0),
)


def reactome(
    source: dict[str, str],
    destination: dict[str, str],
    settings: dict[str, Any],
    config: Config,
) -> None:
    """Generate and post-process Reactome evidence.

    Args:
        source: `evidence`, `disease`, `target` and `literature` locations.
        destination: `evidence` and `failed_evidence` output directories.
        settings: unused.
        config: unused.
    """
    run_evidence(
        scan_dataset(source['evidence'], format='ndjson', infer_schema_length=None),
        REACTOME,
        disease=source['disease'],
        target=source['target'],
        literature=source['literature'],
        destination=destination,
    )
