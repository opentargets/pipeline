"""Clinical report dataset generation."""

import os
from enum import StrEnum
from functools import lru_cache

import polars as pl
import torch
from clinical_mining.data_sources.aact import extract_clinical_report as extract_aact_clinical_report
from clinical_mining.data_sources.chembl.drug_warnings import (
    extract_clinical_report as extract_drug_warning_clinical_report,
)
from clinical_mining.data_sources.chembl.indications import extract_clinical_report as extract_chembl_clinical_report
from clinical_mining.data_sources.ema import extract_clinical_report as extract_ema_clinical_report
from clinical_mining.data_sources.pmda import extract_clinical_report as extract_pmda_clinical_report
from clinical_mining.data_sources.pmda import parse_pmda_approvals
from clinical_mining.data_sources.ttd import extract_clinical_report as extract_ttd_clinical_report
from clinical_mining.dataset import ClinicalReport
from clinical_mining.schemas import ClinicalStageCategory
from clinical_mining.utils.polars_helpers import filter_df, union_dfs
from clinical_mining.utils.spark_helpers import spark_session
from loguru import logger
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from pts.transformers.utils import update_quality_flag


class ClinicalReportFlags(StrEnum):
    PHASE_IV_NOT_APPROVED = 'PHASE_IV_NOT_APPROVED'
    HETEROGENEOUS_DISEASES = 'HETEROGENEOUS_DISEASES'
    UNVALIDATED_INDICATION = 'UNVALIDATED_INDICATION'


def clinical_report(
    source: dict[str, str], destination: dict[str, str], settings: dict[str, str], properties: dict[str, str]
) -> None:
    """Generate clinical report dataset from ChEMBL molecules and disease data.

    Args:
        source: Dictionary containing paths to input data
        destination: Dictionary containing paths to output data:
            - output: Path to write clinical reports (output/clinical_report)
        settings: Dictionary containing step variables
        properties: Dictionary containing Spark properties
    """
    logger.info(f'source paths: {source}')
    spark = spark_session()

    molecule_index_spark = spark.read.parquet(source['chembl_molecule'])
    disease_index_spark = spark.read.parquet(source['disease'])
    chembl_curation = pl.read_parquet(source['chembl_curation']) if 'chembl_curation' in source else None
    aact_studies = pl.read_parquet(source['aact_studies']).select(
        'nct_id',
        'overall_status',
        'phase',
        'study_type',
        'start_date',
        'why_stopped',
        'number_of_arms',
        'official_title',
    )
    aact_interventions = pl.read_parquet(source['aact_interventions']).select(
        'nct_id',
        'intervention_type',
        'name',
    )
    aact_conditions = pl.read_parquet(source['aact_conditions']).select('nct_id', 'downcase_name')
    aact_study_references = pl.read_parquet(source['aact_study_references']).select('nct_id', 'pmid', 'reference_type')
    aact_designs = pl.read_parquet(source['aact_designs']).select('nct_id', 'primary_purpose')
    aact_summaries = pl.read_parquet(source['aact_summaries']).select('nct_id', 'description')
    chembl_indication = pl.read_parquet(source['chembl_indication']).select(
        'drugind_id', 'molregno', 'max_phase_for_ind', 'efo_id', 'efo_term'
    )
    chembl_indication_references = pl.read_parquet(source['chembl_indication_references']).select(
        'drugind_id', 'ref_type', 'ref_id', 'ref_url'
    )
    chembl_molecule_dictionary = pl.read_parquet(source['chembl_molecule_dictionary']).select(
        'molregno', 'chembl_id', 'pref_name'
    )
    chembl_drug_warning = pl.read_parquet(source['chembl_drug_warning']).select(
        'warning_id', 'molregno', 'warning_type', 'warning_year', 'warning_country', 'efo_id', 'efo_term'
    )
    chembl_drug_warning_references = pl.read_parquet(source['chembl_drug_warning_references']).select(
        'warning_id', 'ref_type', 'ref_id', 'ref_url'
    )

    logger.info('extract clinical report')
    pmda = extract_pmda_clinical_report(df=parse_pmda_approvals(pmda_path=source['pmda']), spark=spark)
    aact = extract_aact_clinical_report(
        studies=aact_studies,
        interventions=aact_interventions,
        conditions=aact_conditions,
        additional_metadata=[aact_study_references, aact_designs, aact_summaries],
        aggregation_specs={'pmid': {'group_by': 'nct_id', 'alias': 'literature'}},
    )
    aact_stop_reasons = aact.df.select('id', 'trialWhyStopped').filter(pl.col('trialWhyStopped').is_not_null())
    if aact_stop_reasons.height > 0:
        logger.info(f'categorise stop reasons... input rows: {aact_stop_reasons.height}')
        predictions = predict_trial_stop_reasons(aact_stop_reasons['trialWhyStopped'].to_list())
        stop_reason_predictions = aact_stop_reasons.select('id').with_columns(
            trialStopReasonCategories=pl.Series('trialStopReasonCategories', predictions)
        )
        aact = ClinicalReport(df=aact.df.join(stop_reason_predictions, on='id', how='left'))
        logger.info('categorise stop reasons... complete')
    else:
        aact = ClinicalReport(df=aact.df.with_columns(trialStopReasonCategories=pl.lit(None, dtype=pl.List(pl.String))))
    chembl_indication = extract_chembl_clinical_report(
        drug_indication=chembl_indication,
        molecule_dictionary=chembl_molecule_dictionary,
        indication_refs=chembl_indication_references,
    ).pipe(lambda self: ClinicalReport(filter_df(self.df, "source != 'ClinicalTrials'")))
    chembl_drug_warning = extract_drug_warning_clinical_report(
        drug_warning=chembl_drug_warning,
        molecule_dictionary=chembl_molecule_dictionary,
        warning_refs=chembl_drug_warning_references,
    )
    ttd = extract_ttd_clinical_report(indications_path=source['ttd'])
    ema = extract_ema_clinical_report(indications_path=source['ema'], spark=spark)

    reports = union_dfs([pmda.df, aact.df, chembl_indication.df, chembl_drug_warning.df, ttd.df, ema.df])
    logger.info('reports generated. map entities...')
    output = (
        ClinicalReport
        .map_entities(
            spark=spark,
            reports=reports,
            disease_index=disease_index_spark,
            drug_index=molecule_index_spark,
            chembl_curation=chembl_curation,
            drug_column_name='drugFromSource',
            disease_column_name='diseaseFromSource',
            ner_extract_drug=True,
            ner_batch_size=int(settings['ner_batch_size']),
            ner_cache_path=source['ner_cache_path'],
        )
        .pipe(validate_disease, disease_index=pl.read_parquet(source['disease']))
        .pipe(create_title)
        .pipe(flag_phase_iv_not_approved)
        .pipe(flag_unvalidated_report)
    )

    logger.info(f'destination paths: {destination}')
    output.df.write_parquet(destination['output'], mkdir=True)


def validate_disease(reports: ClinicalReport, disease_index: pl.DataFrame) -> ClinicalReport:
    """Validate disease entities in the reports.

    Args:
        reports: ClinicalReport object with mapped entities
        disease_index: Polars DataFrame with disease index

    Returns:
        ClinicalReport object with validated disease entities
    """
    exploded = reports.df.explode('diseases').unnest('diseases')
    diseases = disease_index.select(pl.col('id').alias('diseaseId'), 'obsoleteTerms').explode('obsoleteTerms')

    # Find valid IDs (those that exist in diseases dataframe or are null in the first place)
    null_ids = exploded.filter(pl.col('diseaseId').is_null())
    non_null = exploded.filter(pl.col('diseaseId').is_not_null())
    valid_ids = non_null.join(diseases.select('diseaseId'), on='diseaseId', how='semi')

    obsolete_ids = non_null.join(
        diseases.select('diseaseId'),
        on='diseaseId',
        how='anti',  # Keep only rows where diseaseId doesn't exist in diseases
    )
    logger.info(f'obsolete disease id count: {obsolete_ids.height}')
    updated_ids = (
        obsolete_ids
        .join(diseases.select('diseaseId', 'obsoleteTerms'), left_on='diseaseId', right_on='obsoleteTerms', how='left')
        .with_columns([
            # Replace obsolete ID with current term, or null if not found
            pl
            .when(pl.col('diseaseId_right').is_not_null())
            .then(pl.col('diseaseId_right'))
            .otherwise(None)
            .alias('diseaseId'),
        ])
        .drop('diseaseId_right')
    )

    combined = pl.concat([null_ids, valid_ids, updated_ids])
    return ClinicalReport(
        df=(
            # Reconstruct the nested disease structure
            combined.group_by(reports.df.drop('diseases').columns, maintain_order=True).agg([
                pl.struct(['diseaseFromSource', 'diseaseId']).unique().alias('diseases')
            ])
        )
    )


def create_title(reports: ClinicalReport) -> ClinicalReport:
    """Create a title for the report based on the information on drugs and diseases."""
    return ClinicalReport(
        df=(
            reports.df
            .with_columns(
                drugs_count=pl.col('drugs').list.len(),
                diseases_count=pl.col('diseases').list.len(),
            )
            .with_columns(
                # Building blocks for the report description
                _stage=pl.col('clinicalStage').str.to_titlecase().str.replace_all('_', ' '),
                _drug=pl.col('drugs').list.first().struct.field('drugFromSource').str.to_titlecase(),
                _disease=pl.col('diseases').list.first().struct.field('diseaseFromSource').str.to_titlecase(),
                _drug_part=pl
                .when(pl.col('drugs_count') == 1)
                .then(pl.col('drugs').list.first().struct.field('drugFromSource').str.to_titlecase())
                .otherwise(pl.concat_str(pl.col('drugs_count').cast(pl.String), pl.lit(' molecules'))),
                _disease_part=pl
                .when(pl.col('diseases_count') == 1)
                .then(pl.col('diseases').list.first().struct.field('diseaseFromSource').str.to_titlecase())
                .otherwise(pl.concat_str(pl.col('diseases_count').cast(pl.String), pl.lit(' diseases'))),
            )
            .with_columns(
                title=pl
                .when(pl.col('trialOfficialTitle').is_not_null())
                .then(pl.col('trialOfficialTitle'))
                .otherwise(
                    pl.concat_str(
                        pl.lit('Report in '),
                        pl.col('_stage'),
                        pl.lit(' stage for '),
                        pl.col('_drug_part'),
                        pl.lit(' and '),
                        pl.col('_disease_part'),
                        ignore_nulls=True,
                    )
                )
            )
            .drop(
                'drugs_count',
                'diseases_count',
                '_stage',
                '_drug',
                '_disease',
                '_drug_part',
                '_disease_part',
            )
        )
    )


# Flagging functions


def flag_phase_iv_not_approved(reports: ClinicalReport) -> ClinicalReport:
    """Flag Phase IV trials that are not approved.

    Args:
        reports: ClinicalReport object with clinical reports

    Returns:
        ClinicalReport object with qualityFlag column updated
    """
    exploded = (
        reports.df
        .explode('drugs')
        .explode('diseases')
        .with_columns([
            pl.col('drugs').struct.field('drugId'),
            pl.col('diseases').struct.field('diseaseId'),
        ])
    )

    approved_pairs = (
        exploded
        .filter(
            pl.col('clinicalStage').is_in([
                ClinicalStageCategory.APPROVAL.value,
                ClinicalStageCategory.PREAPPROVAL.value,
            ])
        )
        .filter(pl.col('drugId').is_not_null() & pl.col('diseaseId').is_not_null())
        .select(['drugId', 'diseaseId'])
        .unique()
    )

    phase_iv_flagged_ids = (
        exploded
        .filter(pl.col('clinicalStage') == ClinicalStageCategory.PHASE_4)
        .filter(pl.col('drugId').is_not_null() & pl.col('diseaseId').is_not_null())
        .select(['id', 'drugId', 'diseaseId'])
        .unique()
        .join(approved_pairs, on=['drugId', 'diseaseId'], how='anti')
        .select('id')
        .unique()
    )

    return ClinicalReport(
        df=update_quality_flag(
            df=reports.df,
            flag_condition=(
                # Flag consists of Phase IV trials that are not approved
                pl.col('clinicalStage').eq(ClinicalStageCategory.PHASE_4)
                & pl.col('id').is_in(phase_iv_flagged_ids['id'].to_list())
            ),
            flag_text=ClinicalReportFlags.PHASE_IV_NOT_APPROVED.value,
        )
    )


def flag_unvalidated_report(reports: ClinicalReport) -> ClinicalReport:
    """Adds flag to reports where any of the indications hasn't been seen in a simple study.

    Simple study: report with a single reported condition and drug
    """
    high_confidence_indications = (
        reports.df
        .filter(
            (pl.col('diseases').list.len() == 1) & (pl.col('drugs').list.len() == 1)
            | pl.col('clinicalStage').is_in([ClinicalStageCategory.APPROVAL, ClinicalStageCategory.PREAPPROVAL])
        )
        .explode('diseases')
        .explode('drugs')
        .with_columns(pl.col('diseases').struct.field('diseaseId'), pl.col('drugs').struct.field('drugId'))
        .filter((pl.col('drugId').is_not_null()) & (pl.col('diseaseId').is_not_null()))
    )

    # Flag reports where the drug/disease pair is NOT part of the high confidence set
    flagged_ids = (
        reports.df
        .explode('diseases')
        .explode('drugs')
        .with_columns(pl.col('diseases').struct.field('diseaseId'), pl.col('drugs').struct.field('drugId'))
        .filter((pl.col('drugId').is_not_null()) & (pl.col('diseaseId').is_not_null()))
        .select(['id', 'diseaseId', 'drugId'])
        .join(high_confidence_indications, on=['diseaseId', 'drugId'], how='anti')
        .select('id')
        .unique()
    )

    return ClinicalReport(
        df=update_quality_flag(
            df=reports.df,
            flag_condition=(
                # Flag consists of reports where any of the indications has not been validated
                pl.col('id').is_in(flagged_ids['id'].to_list())
            ),
            flag_text=ClinicalReportFlags.UNVALIDATED_INDICATION.value,
        )
    )


# STOP REASONS UTILS


def get_device() -> torch.device:
    """Return the best available torch device (MPS > CUDA > CPU)."""
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


@lru_cache(maxsize=1)
def _load_model_assets(model_name: str) -> tuple[AutoTokenizer, AutoModelForSequenceClassification, dict[int, str]]:
    """Load the tokenizer, model, and label mapping — only once, the model is cached."""
    cache_dir = os.environ.get('TRANSFORMERS_CACHE') or os.environ.get('HF_HOME')

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir=cache_dir)
    model.to(get_device())
    model.eval()

    id2label = {int(k): v for k, v in model.config.id2label.items()}

    return tokenizer, model, id2label


def _predict_batch(
    texts: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModelForSequenceClassification,
    id2label: dict[int, str],
    threshold: float,
) -> list[list[str]]:
    """Run inference on a single batch and return predicted labels per text."""
    device = next(model.parameters()).device

    encoded = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    encoded = {k: v.to(device) for k, v in encoded.items()}

    logits = model(**encoded).logits
    probs = torch.sigmoid(logits).cpu()

    results = []
    for prob_row in probs.tolist():
        labels = [id2label[i] for i, prob in enumerate(prob_row) if prob >= threshold]
        results.append(labels or ['Uncategorised'])

    return results


def predict_trial_stop_reasons(
    texts: list[str],
    *,
    model_name: str = 'opentargets/clinical_trial_stop_reasons',
    threshold: float = 0.3,
    batch_size: int = 32,
) -> list[list[str]]:
    """Classify clinical trial stop reasons for a list of texts.

    Args:
        texts:      Input texts to classify.
        model_name: Stop reasons classifier in HuggingFace (default 'opentargets/clinical_trial_stop_reasons').
        threshold:  Minimum sigmoid probability to assign a label (default 0.3).
        batch_size: Number of texts processed per forward pass (default 32).

    Returns:
        A list of predicted label lists, one per input text.
        Texts with no label above the threshold are assigned ['Uncategorised'].
    """
    tokenizer, model, id2label = _load_model_assets(model_name)
    results = []

    with torch.inference_mode():
        for batch_start in range(0, len(texts), batch_size):
            batch_texts = texts[batch_start : batch_start + batch_size]
            batch_results = _predict_batch(batch_texts, tokenizer, model, id2label, threshold)
            results.extend(batch_results)

    return results
