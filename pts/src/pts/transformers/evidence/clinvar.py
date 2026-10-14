"""ClinVar germline evidence, from EVA."""

from __future__ import annotations

from typing import Any

import polars as pl
from otter.config.model import Config

from pts.transformers.evidence.utils.datasource import EvidenceDatasource
from pts.transformers.evidence.utils.expressions import confidence_score
from pts.transformers.evidence.utils.luts import build_luts
from pts.transformers.evidence.utils.run import write_evidence
from pts.transformers.utils.dataset import scan_dataset

SIGNIFICANCE_SCORES = {
    'association not found': 0.0,
    'benign': 0.0,
    'not provided': 0.0,
    'likely benign': 0.0,
    'evidence only': 0.0,
    'likely risk allele': 0.3,
    'low penetrance': 0.3,
    'conflicting interpretations of pathogenicity': 0.3,
    'conflicting data from submitters': 0.3,
    'other': 0.3,
    'uncertain significance': 0.3,
    'uncertain risk allele': 0.3,
    'established risk allele': 0.5,
    'risk factor': 0.5,
    'affects': 0.5,
    'likely pathogenic': 0.7,
    'confers sensitivity': 0.9,
    'association': 0.9,
    'drug response': 0.9,
    'protective': 0.9,
    'pathogenic': 0.9,
}
CONFIDENCE_SCORES = {
    'practice guideline': 0.1,
    'reviewed by expert panel': 0.07,
    'criteria provided, multiple submitters, no conflicts': 0.05,
    'criteria provided, conflicting interpretations': 0.02,
    'criteria provided, single submitter': 0.02,
    'no assertion for the individual variant': 0.0,
    'no assertion criteria provided': 0.0,
    'no assertion provided': 0.0,
}
#: `variantFunctionalConsequenceId` values that map to `'LoF'` in the eva direction_on_target CASE.
LOF_CONSEQUENCES = [
    'SO_0001589',
    'SO_0001587',
    'SO_0001574',
    'SO_0001575',
    'SO_0002012',
    'SO_0001578',
    'SO_0001893',
]


def _score() -> pl.Expr:
    """ClinVar score: best clinical-significance score plus a confidence bonus.

    ```
    coalesce(array_max(transform(clinicalSignificances, x -> element_at(map(...), x))), 0.0)
    + coalesce(element_at(map(...), confidence), 0.0)
    ```

    `list.max()` on an empty or all-unmatched-elements list, and on a null list, is null in polars
    just as spark's `array_max` is -- `fill_null(0.0)` reproduces the `coalesce(..., 0.0)` wrapping
    either way.
    """
    per_significance = (
        pl.col('clinicalSignificances')
        .list.eval(pl.element().replace_strict(SIGNIFICANCE_SCORES, default=None, return_dtype=pl.Float64))
        .list.max()
        .fill_null(0.0)
    )
    confidence_bonus = confidence_score(CONFIDENCE_SCORES).fill_null(0.0)
    return (per_significance + confidence_bonus).round(2)


def _direction_on_trait() -> pl.Expr:
    """ClinVar direction_on_trait: pathogenic/protective terms in `clinicalSignificances`.

    ```
    CASE WHEN CONCAT_WS('', clinicalSignificances) RLIKE '(?i)pathogenic'
     AND CONCAT_WS('', clinicalSignificances) RLIKE '(?i)protect' THEN null
     WHEN ... RLIKE '(?i)pathogenic' THEN 'risk'
     WHEN ... RLIKE '(?i)protect' THEN 'protect'
     ELSE null END
    ```

    `CONCAT_WS('', arr)` skips null elements and never itself returns null; a null list here joins
    to `''`, which matches neither pattern -- the same final `null` result the CASE's `ELSE`
    branch would give, so a null `clinicalSignificances` needs no special casing.
    """
    joined = pl.col('clinicalSignificances').fill_null([]).list.eval(pl.element().fill_null('')).list.join('')
    has_pathogenic = joined.str.contains('(?i)pathogenic')
    has_protect = joined.str.contains('(?i)protect')
    return (
        pl.when(has_pathogenic & has_protect)
        .then(None)
        .when(has_pathogenic)
        .then(pl.lit('risk'))
        .when(has_protect)
        .then(pl.lit('protect'))
        .otherwise(None)
    )


def _direction_on_target() -> pl.Expr:
    """ClinVar direction_on_target: LoF-tagged `variantFunctionalConsequenceId`."""
    return (
        pl.when(pl.col('variantFunctionalConsequenceId').is_in(LOF_CONSEQUENCES))
        .then(pl.lit('LoF'))
        .otherwise(None)
    )


SCORE = _score()
DIRECTION_ON_TRAIT = _direction_on_trait()
DIRECTION_ON_TARGET = _direction_on_target()

EVA = EvidenceDatasource(
    datasource_id='eva',
    # order feeds the id, as in the spark config
    unique_fields=['targetId', 'targetFromSourceId', 'diseaseId', 'datasourceId', 'studyId', 'variantId'],
    score=SCORE,
)


def generate(evidence: str) -> pl.LazyFrame:
    """Read EVA's evidence.

    Args:
        evidence: location of the EVA evidence json.

    Returns:
        Raw EVA evidence.
    """
    return scan_dataset(evidence, format='ndjson', infer_schema_length=None)


def run_eva(source: dict[str, str], destination: dict[str, str], datasource: EvidenceDatasource) -> None:
    """Post-process EVA evidence for one of its datasources and write it.

    Args:
        source: `evidence`, `disease`, `target` and `literature` locations.
        destination: `evidence` and `failed_evidence` output directories.
        datasource: the EVA datasource to keep.
    """
    raw = generate(source['evidence'])
    luts = build_luts(raw, disease=source['disease'], target=source['target'], literature=source['literature'])
    evidence = (
        datasource
        .postprocess(raw, luts)
        .assign_direction_on_trait(DIRECTION_ON_TRAIT)
        .assign_direction_on_target(DIRECTION_ON_TARGET, luts.target)
        .hash_long_variant_identifiers()
    )
    write_evidence(evidence, destination)


def clinvar(
    source: dict[str, str],
    destination: dict[str, str],
    settings: dict[str, Any],
    config: Config,
) -> None:
    """Generate and post-process ClinVar germline evidence.

    Args:
        source: `evidence`, `disease`, `target` and `literature` locations.
        destination: `evidence` and `failed_evidence` output directories.
        settings: unused.
        config: unused.
    """
    run_eva(source, destination, EVA)
