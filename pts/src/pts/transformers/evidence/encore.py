"""ENCORE genetic interaction evidence."""

from __future__ import annotations

from typing import Any

import polars as pl
from otter.config.model import Config

from pts.schemas.outputs.evidence_encore import EvidenceEncore
from pts.transformers.evidence.utils.datasource import EvidenceDatasource
from pts.transformers.evidence.utils.run import run_evidence
from pts.transformers.utils.dataset import scan_dataset
from pts.transformers.utils.spark_math import divide

ENCORE = EvidenceDatasource(
    datasource_id='encore',
    # order and repetition feed the id, as in the spark config
    unique_fields=[
        'targetId',
        'targetFromSourceId',
        'diseaseId',
        'datasourceId',
        'targetFromSourceId',
        'interactingTargetFromSourceId',
        'diseaseFromSourceMappedId',
        'diseaseCellLines',
    ],
    score=divide(pl.col('geneticInteractionScore').cast(pl.Float64).abs(), 16.0),
    schema=EvidenceEncore,
)


def encore(
    source: dict[str, str],
    destination: dict[str, str],
    settings: dict[str, Any],
    config: Config,
) -> None:
    """Generate and post-process ENCORE evidence.

    Args:
        source: `evidence`, `disease` and `target` locations.
        destination: `evidence` and `failed_evidence` output directories.
        settings: unused.
        config: unused.
    """
    run_evidence(
        scan_dataset(source['evidence']),
        ENCORE,
        disease=source['disease'],
        target=source['target'],
        destination=destination,
    )
