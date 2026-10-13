"""UniProt literature evidence."""

from __future__ import annotations

from typing import Any

import polars as pl
from otter.config.model import Config

from pts.transformers.evidence.uniprot_variants import CONFIDENCE_SCORES
from pts.transformers.evidence.utils.datasource import EvidenceDatasource
from pts.transformers.evidence.utils.expressions import confidence_score
from pts.transformers.evidence.utils.run import run_evidence
from pts.transformers.utils.dataset import scan_dataset

UNIPROT_LITERATURE = EvidenceDatasource(
    datasource_id='uniprot_literature',
    # order feeds the id, as in the spark config
    unique_fields=['targetId', 'targetFromSourceId', 'diseaseId', 'datasourceId', 'diseaseFromSource'],
    score=confidence_score(CONFIDENCE_SCORES),
)


def generate(evidence: str) -> pl.LazyFrame:
    """Read UniProt literature evidence.

    Args:
        evidence: location of the UniProt literature evidence json.

    Returns:
        Raw UniProt literature evidence.
    """
    return scan_dataset(evidence, format='ndjson', infer_schema_length=None)


def uniprot_literature(
    source: dict[str, str],
    destination: dict[str, str],
    settings: dict[str, Any],
    config: Config,
) -> None:
    """Generate and post-process UniProt literature evidence.

    Args:
        source: `evidence`, `disease`, `target` and `literature` locations.
        destination: `evidence` and `failed_evidence` output directories.
        settings: unused.
        config: unused.
    """
    run_evidence(
        generate(source['evidence']),
        UNIPROT_LITERATURE,
        disease=source['disease'],
        target=source['target'],
        literature=source['literature'],
        destination=destination,
    )
