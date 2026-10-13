"""ClinVar somatic evidence, from EVA."""

from __future__ import annotations

from typing import Any

from otter.config.model import Config

from pts.transformers.evidence.clinvar import EVA, run_eva
from pts.transformers.evidence.utils.datasource import EvidenceDatasource

EVA_SOMATIC = EvidenceDatasource(datasource_id='eva_somatic', unique_fields=EVA.unique_fields, score=EVA.score)


def clinvar_somatic(
    source: dict[str, str],
    destination: dict[str, str],
    settings: dict[str, Any],
    config: Config,
) -> None:
    """Generate and post-process ClinVar somatic evidence.

    Args:
        source: `evidence`, `disease`, `target` and `literature` locations.
        destination: `evidence` and `failed_evidence` output directories.
        settings: unused.
        config: unused.
    """
    run_eva(source, destination, EVA_SOMATIC)
