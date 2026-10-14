"""UniProt variants evidence."""

from __future__ import annotations

from typing import Any

from otter.config.model import Config

from pts.transformers.evidence.utils.datasource import EvidenceDatasource
from pts.transformers.evidence.utils.expressions import confidence_score
from pts.transformers.evidence.utils.luts import build_luts
from pts.transformers.evidence.utils.run import write_evidence
from pts.transformers.utils.dataset import scan_dataset

CONFIDENCE_SCORES = {'high': 1.0, 'medium': 0.5}

UNIPROT_VARIANTS = EvidenceDatasource(
    datasource_id='uniprot_variants',
    # order feeds the id, as in the spark config
    unique_fields=[
        'targetId',
        'targetFromSourceId',
        'diseaseId',
        'datasourceId',
        'diseaseFromSource',
        'variantRsId',
        'variantId',
        'literature',
    ],
    score=confidence_score(CONFIDENCE_SCORES),
)


def uniprot_variants(
    source: dict[str, str],
    destination: dict[str, str],
    settings: dict[str, Any],
    config: Config,
) -> None:
    """Generate and post-process UniProt variants evidence.

    Args:
        source: `evidence`, `disease`, `target` and `literature` locations.
        destination: `evidence` and `failed_evidence` output directories.
        settings: unused.
        config: unused.
    """
    raw = scan_dataset(source['evidence'], format='ndjson', infer_schema_length=None)
    luts = build_luts(raw, disease=source['disease'], target=source['target'], literature=source['literature'])
    write_evidence(UNIPROT_VARIANTS.postprocess(raw, luts).hash_long_variant_identifiers(), destination)
