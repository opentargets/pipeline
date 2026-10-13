"""Validation Lab (ot_crispr_validation) evidence."""

from __future__ import annotations

from typing import Any

import polars as pl
from otter.config.model import Config

from pts.transformers.evidence.utils.datasource import EvidenceDatasource
from pts.transformers.evidence.utils.run import run_evidence
from pts.transformers.utils.dataset import scan_dataset

OT_CRISPR_VALIDATION = EvidenceDatasource(
    datasource_id='ot_crispr_validation',
    # order and repetition feed the id, as in the spark config
    unique_fields=[
        'targetId',
        'targetFromSourceId',
        'diseaseId',
        'datasourceId',
        'targetFromSourceId',
        'diseaseFromSourceMappedId',
        'resourceScore',
        'diseaseCellLines',
        'primaryProjectId',
    ],
    score=pl.col('resourceScore'),
)


def generate(evidence: str) -> pl.LazyFrame:
    """Read Validation Lab evidence.

    Args:
        evidence: location of the Validation Lab evidence json.

    Returns:
        Raw Validation Lab evidence.
    """
    return scan_dataset(evidence, format='ndjson', infer_schema_length=None)


def validation_lab(
    source: dict[str, str],
    destination: dict[str, str],
    settings: dict[str, Any],
    config: Config,
) -> None:
    """Generate and post-process Validation Lab evidence.

    Args:
        source: `evidence`, `disease`, `target` and `literature` locations.
        destination: `evidence` and `failed_evidence` output directories.
        settings: unused.
        config: unused.
    """
    run_evidence(
        generate(source['evidence']),
        OT_CRISPR_VALIDATION,
        disease=source['disease'],
        target=source['target'],
        literature=source['literature'],
        destination=destination,
    )
