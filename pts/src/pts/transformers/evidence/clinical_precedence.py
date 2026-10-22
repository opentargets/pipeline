"""Clinical precedence evidence, from clinical reports and drug mechanisms of action."""

from __future__ import annotations

from typing import Any

import polars as pl
from loguru import logger
from otter.config.model import Config

from pts.schemas.outputs.evidence_clinical_precedence import EvidenceClinicalPrecedence
from pts.transformers.evidence.utils.datasource import EvidenceDatasource
from pts.transformers.evidence.utils.expressions import direction_when_disease_present
from pts.transformers.evidence.utils.luts import build_luts
from pts.transformers.evidence.utils.run import write_evidence
from pts.transformers.utils.dataset import scan_dataset, sink_dataset

TRIAL_OUTCOME_REFERENCE_TYPES = ['result', 'derived']

#: `clinicalStage` scores.
CLINICAL_STAGE_SCORES = {
    'APPROVAL': 1.0,
    'PHASE_4': 1.0,
    'WITHDRAWAL': 1.0,
    'PREAPPROVAL': 0.8,
    'PHASE_3': 0.7,
    'PHASE_2_3': 0.5,
    'PHASE_2': 0.2,
    'PHASE_1_2': 0.15,
    'PHASE_1': 0.1,
    'EARLY_PHASE_1': 0.05,
    'IND': 0.05,
    'PRECLINICAL': 0.01,
    'UNKNOWN': 0.01,
}
#: `trialStopReasonCategories` element multipliers.
TRIAL_STOP_REASON_SCORES = {
    'Another_Study': 1.0,
    'Business_Administrative': 1.0,
    'Covid19': 1.0,
    'Ethical_Reason': 1.0,
    'Insufficient_Data': 1.0,
    'Insufficient_Enrollment': 1.0,
    'Interim_Analysis': 1.0,
    'Invalid_Reason': 1.0,
    'Logistics_Resources': 1.0,
    'Endpoint_Met': 1.0,
    'Negative': 0.5,
    'No_Context': 1.0,
    'Regulatory': 1.0,
    'Safety_Sideeffects': 0.5,
    'Study_Design': 1.0,
    'Study_Staff_Moved': 1.0,
    'Success': 1.0,
    'Uncategorised': 1.0,
}
#: `actionType` values that map to `'LoF'` / `'GoF'`.
LOF_ACTION_TYPES = [
    'RNAI INHIBITOR',
    'NEGATIVE MODULATOR',
    'NEGATIVE ALLOSTERIC MODULATOR',
    'ANTAGONIST',
    'ANTISENSE INHIBITOR',
    'BLOCKER',
    'INHIBITOR',
    'DEGRADER',
    'INVERSE AGONIST',
    'ALLOSTERIC ANTAGONIST',
    'DISRUPTING AGENT',
    'GENE EDITING NEGATIVE MODULATOR',
]
GOF_ACTION_TYPES = [
    'PARTIAL AGONIST',
    'ACTIVATOR',
    'POSITIVE ALLOSTERIC MODULATOR',
    'POSITIVE MODULATOR',
    'AGONIST',
    'SEQUESTERING AGENT',
    'STABILISER',
]


def _score() -> pl.Expr:
    """Clinical-stage score times the lowest trial-stop-reason multiplier.

    A null `trialStopReasonCategories` multiplies by 1.0. `list.min()` skips unmatched (null)
    elements and, like spark's `array_min`, is null on an empty list, so an empty list scores null.
    """
    stage_score = pl.col('clinicalStage').replace_strict(CLINICAL_STAGE_SCORES, default=None, return_dtype=pl.Float64)
    stop_multiplier = (
        pl
        .when(pl.col('trialStopReasonCategories').is_null())
        .then(pl.lit(1.0))
        .otherwise(
            pl
            .col('trialStopReasonCategories')
            .list.eval(pl.element().replace_strict(TRIAL_STOP_REASON_SCORES, default=None, return_dtype=pl.Float64))
            .list.min()
        )
    )
    return stage_score * stop_multiplier


def _direction_on_target() -> pl.Expr:
    """LoF or GoF from `actionType`."""
    action_type = pl.col('actionType')
    return (
        pl
        .when(action_type.is_in(LOF_ACTION_TYPES))
        .then(pl.lit('LoF'))
        .when(action_type.is_in(GOF_ACTION_TYPES))
        .then(pl.lit('GoF'))
        .otherwise(None)
    )


CLINICAL_PRECEDENCE = EvidenceDatasource(
    datasource_id='clinical_precedence',
    unique_fields=['targetId', 'targetFromSourceId', 'diseaseId', 'datasourceId', 'drugId', 'clinicalReportId'],
    score=_score(),
    schema=EvidenceClinicalPrecedence,
)
DIRECTION_ON_TRAIT = direction_when_disease_present('protect')
DIRECTION_ON_TARGET = _direction_on_target()


def result_literature(trial_literature: pl.Expr) -> pl.Expr:
    """Flatten `trialLiterature` structs to the PMIDs that report the trial's outcome.

    Background citations are dropped, as are entries with no PMID; reference types match
    case-insensitively. Null when nothing survives the filter.
    """
    reference = pl.element()
    pmids = trial_literature.list.eval(
        reference.filter(
            reference.struct.field('type').str.to_lowercase().is_in(TRIAL_OUTCOME_REFERENCE_TYPES)
            & reference.struct.field('id').is_not_null()
        ).struct.field('id')
    )
    return pl.when(pmids.list.len() > 0).then(pmids).otherwise(None)


def _explode(frame: pl.LazyFrame, column: str) -> pl.LazyFrame:
    """Explode as spark's `explode` does, dropping rows whose list is null or empty."""
    return frame.explode(column, empty_as_null=False, keep_nulls=False)


def generate(
    clinical_report: str,
    drug_mechanism_of_action: str,
    invalid_clinical_report_qc: list[str],
) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """Build clinical precedence evidence by joining clinical reports with drug mechanisms of action.

    Args:
        clinical_report: location of the clinical report dataset.
        drug_mechanism_of_action: location of the drug mechanism of action dataset.
        invalid_clinical_report_qc: `qualityControls` values that exclude a report.

    Returns:
        Raw clinical precedence evidence, one row per report, disease, drug and target; and the
        clinical reports excluded by quality control.
    """
    reports = scan_dataset(clinical_report)
    has_invalid_qc = (
        pl.col('qualityControls').list.eval(pl.element().is_in(invalid_clinical_report_qc)).list.any().fill_null(False)
    )
    excluded = reports.filter(has_invalid_qc)

    report_evidence = (
        reports
        .filter(~has_invalid_qc)
        .select(
            pl.col('id').alias('clinicalReportId'),
            'clinicalStage',
            'trialWhyStopped',
            'trialStopReasonCategories',
            pl.col('trialStartDate').cast(pl.String).alias('studyStartDate'),
            pl.col('year').cast(pl.String).alias('evidenceDate'),
            result_literature(pl.col('trialLiterature')).alias('literature'),
            'diseases',
            'drugs',
        )
        .pipe(_explode, 'diseases')
        .unnest('diseases')
        .rename({'diseaseId': 'diseaseFromSourceMappedId'})
        .pipe(_explode, 'drugs')
        .unnest('drugs')
        .filter(pl.col('diseaseFromSourceMappedId').is_not_null() & pl.col('drugId').is_not_null())
    )
    mechanisms = (
        scan_dataset(drug_mechanism_of_action)
        .select(pl.col('chemblIds').alias('drugId'), pl.col('targets').alias('targetFromSourceId'), 'actionType')
        .pipe(_explode, 'drugId')
    )
    evidence = (
        report_evidence
        .join(mechanisms, on='drugId', how='inner', maintain_order='left')
        .pipe(_explode, 'targetFromSourceId')
        .filter(pl.col('targetFromSourceId').is_not_null())
        .select(
            'clinicalReportId',
            'clinicalStage',
            'trialWhyStopped',
            'trialStopReasonCategories',
            'studyStartDate',
            'evidenceDate',
            'literature',
            'diseaseFromSource',
            'diseaseFromSourceMappedId',
            'drugFromSource',
            'drugId',
            'targetFromSourceId',
            pl.lit('clinical_precedence').alias('datasourceId'),
            pl.lit('clinical').alias('datatypeId'),
            'actionType',
        )
        .unique(maintain_order=True)
    )
    return evidence, excluded


def clinical_precedence(
    source: dict[str, str],
    destination: dict[str, str],
    settings: dict[str, Any],
    config: Config,
) -> None:
    """Generate and post-process clinical precedence evidence, and write the excluded reports.

    Args:
        source: `clinical_report`, `drug_mechanism_of_action`, `disease`, `target` and `literature`
            locations.
        destination: `evidence`, `failed_evidence` and `excluded_report` output directories.
        settings: `invalid_clinical_report_qc`, the `qualityControls` values that exclude a report.
        config: unused.
    """
    raw, excluded = generate(
        source['clinical_report'],
        source['drug_mechanism_of_action'],
        settings.get('invalid_clinical_report_qc', []),
    )
    luts = build_luts(raw, disease=source['disease'], target=source['target'], literature=source['literature'])
    evidence = (
        CLINICAL_PRECEDENCE
        .postprocess(raw, luts)
        .assign_direction_on_trait(DIRECTION_ON_TRAIT)
        .assign_direction_on_target(DIRECTION_ON_TARGET, luts.target)
    )
    logger.info(f'writing excluded clinical reports to {destination["excluded_report"]}')
    write_evidence(
        evidence,
        destination,
        alongside=[sink_dataset(excluded, destination['excluded_report'])],
        schema=CLINICAL_PRECEDENCE.schema,
    )
