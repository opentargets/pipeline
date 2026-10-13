"""GWAS credible set evidence, from the locus-to-gene predictions.

Replaces gentropy's `LocusToGeneEvidenceStep` (`L2GPrediction.to_disease_target_evidence`) and the
`gwas_credible_sets` datasource of the generic evidence post-processing. Raw evidence generation
ported from opentargets/pipeline#111.
"""

from __future__ import annotations

from typing import Any

import polars as pl
from otter.config.model import Config

from pts.transformers.evidence.postprocess import EvidencePostprocessor
from pts.transformers.evidence.run import run_evidence
from pts.transformers.utils.dataset import scan_dataset

GWAS_CREDIBLE_SETS = EvidencePostprocessor(
    datasource_id='gwas_credible_sets',
    unique_fields=['targetId', 'targetFromSourceId', 'diseaseId', 'datasourceId', 'studyLocusId'],
    score=pl.col('resourceScore'),
)

#: Substring match, as in gentropy, so `9999-99-99` passes.
_CURATION_DATE = r'\d{4}-\d{2}-\d{2}'


def build_raw_evidence(
    predictions: pl.LazyFrame,
    credible_set: pl.LazyFrame,
    study_index: pl.LazyFrame,
    threshold: float,
) -> pl.LazyFrame:
    """Turn L2G predictions into one raw evidence row per prediction and study disease.

    Args:
        predictions: `studyLocusId`, `geneId`, `score`.
        credible_set: `studyLocusId`, `studyId`.
        study_index: `studyId`, `diseaseIds`, `pubmedId`, `publicationDate`.
        threshold: minimum `score` for a prediction to become evidence.

    Returns:
        Raw GWAS credible set evidence.
    """
    studies = study_index.select(
        'studyId',
        'diseaseIds',
        pl
        .when(pl.col('publicationDate').str.contains(_CURATION_DATE))
        .then(pl.col('publicationDate'))
        .alias('curationDate'),
        pl.when(pl.col('pubmedId').is_not_null()).then(pl.concat_list(pl.col('pubmedId'))).alias('literature'),
    )
    return (
        predictions
        .filter(pl.col('score') >= threshold)
        .join(credible_set, on='studyLocusId', how='inner')
        .join(studies, on='studyId', how='inner')
        .filter(pl.col('diseaseIds').list.len() > 0)
        .explode('diseaseIds')
        .select(
            pl.lit('genetic_association').alias('datatypeId'),
            pl.lit('gwas_credible_sets').alias('datasourceId'),
            pl.col('geneId').alias('targetFromSourceId'),
            pl.col('diseaseIds').alias('diseaseFromSourceMappedId'),
            pl.col('score').alias('resourceScore'),
            'curationDate',
            'studyLocusId',
            'literature',
        )
    )


def generate(l2g_predictions: str, credible_set: str, study_index: str, threshold: float) -> pl.LazyFrame:
    """Read the L2G inputs and build raw GWAS credible set evidence.

    Args:
        l2g_predictions: location of the L2G predictions.
        credible_set: location of the credible sets.
        study_index: location of the study index.
        threshold: minimum L2G score.

    Returns:
        Raw GWAS credible set evidence.
    """
    return build_raw_evidence(
        scan_dataset(l2g_predictions).select('studyLocusId', 'geneId', 'score'),
        scan_dataset(credible_set).select('studyLocusId', 'studyId'),
        scan_dataset(study_index).select('studyId', 'diseaseIds', 'pubmedId', 'publicationDate'),
        threshold,
    )


def gwas_credible_sets(
    source: dict[str, str],
    destination: dict[str, str],
    settings: dict[str, Any],
    config: Config,
) -> None:
    """Generate and post-process GWAS credible set evidence.

    Args:
        source: `l2g_predictions`, `credible_set`, `study_index`, `disease`, `target` and
            `literature` locations.
        destination: `evidence` and `failed_evidence` output directories.
        settings: `locus_to_gene_threshold`.
        config: unused.
    """
    raw = generate(
        source['l2g_predictions'],
        source['credible_set'],
        source['study_index'],
        float(settings['locus_to_gene_threshold']),
    )
    run_evidence(
        raw,
        GWAS_CREDIBLE_SETS,
        disease=source['disease'],
        target=source['target'],
        literature=source['literature'],
        destination=destination,
    )
