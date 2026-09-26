"""Generate and validate GWAS credible-set locus-to-gene evidence.

Merges what used to be two pipeline steps:

* gentropy's `LocusToGeneEvidenceStep` (`gentropy.l2g.LocusToGeneEvidenceStep` /
  `L2GPrediction.to_disease_target_evidence`), which turns L2G gene-disease predictions into raw
  evidence records, and
* the `gwas_credible_sets` datasource of the generic PySpark `evidence_postprocess` step
  (`pts.pyspark.evidence_postprocess`), which validates and dates that raw evidence.

Target protein-coding validation and disease-ID/obsolete-term normalisation both already happened
upstream in gentropy (the L2G feature matrix and study validation steps, respectively) before this
step's inputs were written, so they are not repeated here. What *is* repeated here is the second,
independent disease/target validation against PTS's own disease and target indices -- the same one
every other evidence source goes through.
"""

from pathlib import Path
from typing import Any

import polars as pl
from loguru import logger
from otter.config.model import Config

from pts.schemas.evidence import GwasCredibleSetEvidenceSchema
from pts.transformers.evidence import (
    assign_evidence_identifier,
    build_disease_lut,
    build_publication_lut,
    build_target_lut,
    calculate_evidence_score,
    resolve_evidence_date,
    resolve_publication_date,
    validate_diseases,
    validate_target,
    validate_uniqueness,
)
from pts.transformers.utils.dataset import scan_dataset, write_dataset

_DEFAULT_THRESHOLD = 0.05
#: Loose validity check on a date string, matching gentropy's own -- a substring match, not a
#: real date parse, so e.g. "9999-99-99" passes. Replicated as-is rather than "fixed".
_CURATION_DATE_REGEX = r'\d{4}-\d{2}-\d{2}'


def gwas_evidence(
    source: dict[str, Path],
    destination: dict[str, Path],
    settings: dict[str, Any],
    _config: Config,
) -> None:
    """Build, validate and write GWAS credible-set locus-to-gene evidence.

    Args:
        source: paths to `l2g_predictions`, `credible_set`, `study_index` (gentropy outputs) and
            `disease`, `target`, `literature` (PTS/PIS outputs, for validation/dating).
        destination: `evidence` (valid records) and `failed_evidence` (flagged records) paths.
        settings: `locus_to_gene_threshold` (float, default 0.05) and `unique_fields`
            (list[str], the evidence-identifier key columns).
        _config: otter Config object, unused.
    """
    threshold = settings.get('locus_to_gene_threshold', _DEFAULT_THRESHOLD)
    unique_fields = settings['unique_fields']

    logger.info(f'Reading L2G predictions from {source["l2g_predictions"]}')
    predictions = scan_dataset(str(source['l2g_predictions'])).select('studyLocusId', 'geneId', 'score').collect()
    logger.info(f'Reading credible sets from {source["credible_set"]}')
    credible_set = scan_dataset(str(source['credible_set'])).select('studyLocusId', 'studyId').collect()
    logger.info(f'Reading study index from {source["study_index"]}')
    study_index = (
        scan_dataset(str(source['study_index']))
        .select('studyId', 'diseaseIds', 'pubmedId', 'publicationDate')
        .collect()
    )

    logger.info('Building raw GWAS credible set evidence')
    raw_evidence = _build_raw_evidence(predictions, credible_set, study_index, threshold)

    logger.info(f'Reading disease index from {source["disease"]}')
    disease_lut = build_disease_lut(scan_dataset(str(source['disease'])).select('id', 'obsoleteTerms').collect())
    logger.info(f'Reading target index from {source["target"]}')
    target_lut = build_target_lut(
        scan_dataset(str(source['target'])).select('id', 'biotype', 'proteinIds', 'approvedSymbol').collect()
    )
    logger.info(f'Reading literature export from {source["literature"]}')
    publication_lut = build_publication_lut(
        scan_dataset(str(source['literature']), format='ndjson')
        .select('source', 'firstPublicationDate', 'pmid', 'id', 'pmcid')
        .collect()
    )

    logger.info('Validating and dating GWAS credible set evidence')
    processed = (
        raw_evidence
        .pipe(validate_diseases, disease_lut)
        .pipe(validate_target, target_lut)
        .pipe(assign_evidence_identifier, unique_fields)
        .pipe(validate_uniqueness)
        .pipe(resolve_publication_date, publication_lut)
        .pipe(resolve_evidence_date)
        .pipe(calculate_evidence_score, pl.col('resourceScore'))
    )

    valid = processed.filter(pl.col('qualityControls').list.len() == 0)
    invalid = processed.filter(pl.col('qualityControls').list.len() > 0)

    logger.info(f'Writing {valid.height} valid and {invalid.height} invalid GWAS evidence records')
    write_dataset(valid, str(destination['evidence']), schema=GwasCredibleSetEvidenceSchema)
    write_dataset(invalid, str(destination['failed_evidence']))


def _build_raw_evidence(
    predictions: pl.DataFrame,
    credible_set: pl.DataFrame,
    study_index: pl.DataFrame,
    threshold: float,
) -> pl.DataFrame:
    """Build raw (unvalidated) GWAS evidence from L2G predictions, credible sets and studies.

    Polars port of `L2GPrediction.to_disease_target_evidence`
    (gentropy `src/gentropy/dataset/l2g_prediction.py:113-183`).

    Args:
        predictions: L2G predictions with `studyLocusId`, `geneId`, `score` columns.
        credible_set: credible sets with `studyLocusId`, `studyId` columns.
        study_index: studies with `studyId`, `diseaseIds`, `pubmedId`, `publicationDate` columns.
        threshold: minimum `score` for a prediction to produce evidence.

    Returns:
        One row per (prediction, disease) pair: `datatypeId`, `datasourceId`,
        `targetFromSourceId`, `diseaseFromSourceMappedId`, `resourceScore`, `curationDate`,
        `studyLocusId`, `literature`.
    """
    study_index_slim = study_index.select(
        'studyId',
        'diseaseIds',
        pl
        .when(pl.col('publicationDate').str.contains(_CURATION_DATE_REGEX))
        .then(pl.col('publicationDate'))
        .otherwise(None)
        .alias('curationDate'),
        pl
        .when(pl.col('pubmedId').is_not_null())
        .then(pl.concat_list(pl.col('pubmedId')))
        .otherwise(None)
        .alias('literature'),
    )

    return (
        predictions
        .filter(pl.col('score') >= threshold)
        .join(credible_set, on='studyLocusId', how='inner')
        .join(study_index_slim, on='studyId', how='inner')
        .filter(pl.col('diseaseIds').is_not_null() & (pl.col('diseaseIds').list.len() > 0))
        .explode('diseaseIds', empty_as_null=True)
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
