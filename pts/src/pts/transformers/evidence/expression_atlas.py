"""Expression Atlas evidence."""

from __future__ import annotations

from typing import Any

import polars as pl
from otter.config.model import Config

from pts.schemas.outputs.evidence_expression_atlas import EvidenceExpressionAtlas
from pts.transformers.evidence.utils.datasource import EvidenceDatasource
from pts.transformers.evidence.utils.expressions import log10_rescaled_score
from pts.transformers.evidence.utils.run import run_evidence
from pts.transformers.utils.dataset import scan_dataset
from pts.transformers.utils.spark_math import divide

EXCLUDED_BIOTYPES = [
    'IG_C_pseudogene',
    'IG_J_pseudogene',
    'IG_pseudogene',
    'IG_V_pseudogene',
    'polymorphic_pseudogene',
    'processed_pseudogene',
    'pseudogene',
    'rRNA',
    'rRNA_pseudogene',
    'snoRNA',
    'snRNA',
    'transcribed_processed_pseudogene',
    'transcribed_unitary_pseudogene',
    'transcribed_unprocessed_pseudogene',
    'TR_J_pseudogene',
    'TR_V_pseudogene',
    'unitary_pseudogene',
    'unprocessed_pseudogene',
]


def _score() -> pl.Expr:
    """Expression Atlas score: log10-rescaled p-value scaled by fold-change magnitude and rank.

    ```
    array_min(array(1.0,
      (CASE ... log10 rescale of resourceScore ... END)
      * (abs(log2FoldChangeValue) / 10) * (log2FoldChangePercentileRank / 100)
    ))
    ```

    A null or NaN `resourceScore` makes the product null, which `min_horizontal` skips like spark's
    `array_min`, so the row scores 1.0.
    """
    base = log10_rescaled_score(pl.col('resourceScore'), 0.0, 1.0, 1e-10)
    magnitude = (
        base
        * divide(pl.col('log2FoldChangeValue').abs(), 10)
        * divide(pl.col('log2FoldChangePercentileRank'), 100)
    )
    return pl.min_horizontal(pl.lit(1.0), magnitude)


SCORE = _score()

EXPRESSION_ATLAS = EvidenceDatasource(
    datasource_id='expression_atlas',
    # order feeds the id, as in the spark config
    unique_fields=['targetId', 'targetFromSourceId', 'diseaseId', 'datasourceId', 'contrast', 'studyId'],
    score=SCORE,
    excluded_biotypes=EXCLUDED_BIOTYPES,
    schema=EvidenceExpressionAtlas,
)


def expression_atlas(
    source: dict[str, str],
    destination: dict[str, str],
    settings: dict[str, Any],
    config: Config,
) -> None:
    """Generate and post-process Expression Atlas evidence.

    Args:
        source: `evidence`, `disease`, `target` and `literature` locations.
        destination: `evidence` and `failed_evidence` output directories.
        settings: unused.
        config: unused.
    """
    run_evidence(
        scan_dataset(source['evidence']),
        EXPRESSION_ATLAS,
        disease=source['disease'],
        target=source['target'],
        literature=source['literature'],
        destination=destination,
    )
