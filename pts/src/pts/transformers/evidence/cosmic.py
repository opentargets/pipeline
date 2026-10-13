"""COSMIC Cancer Gene Census evidence."""

from __future__ import annotations

from typing import Any

import polars as pl
from otter.config.model import Config

from pts.transformers.evidence.utils.datasource import EvidenceDatasource
from pts.transformers.evidence.utils.expressions import direction_lookup, direction_when_disease_present
from pts.transformers.evidence.utils.luts import build_luts
from pts.transformers.evidence.utils.run import write_evidence
from pts.transformers.utils.dataset import scan_dataset

CANCER_GENE_CENSUS = EvidenceDatasource(
    datasource_id='cancer_gene_census',
    unique_fields=['targetId', 'targetFromSourceId', 'diseaseId', 'datasourceId'],
    score=pl.col('resourceScore'),
)
DIRECTION_ON_TRAIT = direction_when_disease_present('risk')
DIRECTION_ON_TARGET = direction_lookup('TSorOncogene', {'oncogene': 'GoF', 'tsg': 'LoF'})


def generate(evidence: str) -> pl.LazyFrame:
    """Read COSMIC's evidence.

    Args:
        evidence: location of the COSMIC evidence json.

    Returns:
        Raw COSMIC evidence.
    """
    return scan_dataset(evidence, format='ndjson', infer_schema_length=None)


def cosmic(
    source: dict[str, str],
    destination: dict[str, str],
    settings: dict[str, Any],
    config: Config,
) -> None:
    """Generate and post-process Cancer Gene Census evidence.

    Args:
        source: `evidence`, `disease`, `target` and `literature` locations.
        destination: `evidence` and `failed_evidence` output directories.
        settings: unused.
        config: unused.
    """
    raw = generate(source['evidence'])
    CANCER_GENE_CENSUS.check_raw(raw)
    luts = build_luts(raw, disease=source['disease'], target=source['target'], literature=source['literature'])
    evidence = (
        CANCER_GENE_CENSUS
        .postprocess(raw, luts)
        .assign_direction_on_trait(DIRECTION_ON_TRAIT)
        .assign_direction_on_target(DIRECTION_ON_TARGET, luts.target)
        .hash_long_variant_identifiers()
    )
    write_evidence(evidence, destination)
