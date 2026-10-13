"""ENCORE genetic interaction evidence."""

from __future__ import annotations

from typing import Any

import polars as pl
from otter.config.model import Config

from pts.transformers.evidence.expressions import divide
from pts.transformers.evidence.postprocess import EvidencePostprocessor
from pts.transformers.evidence.run import run_evidence
from pts.transformers.utils.dataset import scan_dataset

ENCORE = EvidencePostprocessor(
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
)


def generate(evidence: str) -> pl.LazyFrame:
    """Read ENCORE's evidence, which arrives in the evidence format.

    Args:
        evidence: location of the ENCORE evidence parquet.

    Returns:
        Raw ENCORE evidence.
    """
    return scan_dataset(evidence)


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
        generate(source['evidence']),
        ENCORE,
        disease=source['disease'],
        target=source['target'],
        destination=destination,
    )
