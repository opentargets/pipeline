"""Clinical report dataset generation."""

from enum import StrEnum
from typing import Any

import polars as pl
from loguru import logger
from otter.config.model import Config

from pts.clinical_staging.staged_outputs import DISEASE_ENTITIES, DRUG_ENTITIES, EXTRACTION, STOP_REASONS
from pts.postgres import read_dump_tables
from pts.transformers.clinical import (
    ClinicalReport,
    ClinicalReportType,
    ClinicalSource,
    ClinicalStageCategory,
    extract_aact_clinical_report,
    extract_chembl_clinical_report,
    extract_drug_warning_clinical_report,
    filter_df,
    map_entities,
    union_dfs,
)
from pts.transformers.clinical.mapping import require_staged
from pts.transformers.clinical.mining import mining_reports, read_mining_inputs, staged_lookup
from pts.transformers.clinical.providers.aact import read_aact_tables
from pts.transformers.ontoma_lut.build import DISEASE_COLUMNS, DRUG_COLUMNS
from pts.transformers.utils import update_quality_flag
from pts.transformers.utils.dataset import scan_dataset, write_dataset

CHEMBL_SCHEMA_NAME = 'public'

CHEMBL_TABLES = {
    'drug_indication': ['drugind_id', 'molregno', 'max_phase_for_ind', 'efo_id', 'efo_term'],
    'indication_refs': ['drugind_id', 'ref_type', 'ref_id', 'ref_url'],
    'molecule_dictionary': ['molregno', 'chembl_id', 'pref_name'],
    'drug_warning': [
        'warning_id',
        'molregno',
        'warning_type',
        'warning_year',
        'warning_country',
        'warning_class',
        'efo_id',
        'efo_term',
        'efo_id_for_warning_class',
    ],
    'warning_refs': ['warning_id', 'ref_type', 'ref_id', 'ref_url'],
}

CHEMBL_ORDER_BY = {
    'indication_refs': ['drugind_id', 'ref_type', 'ref_id', 'ref_url'],
    'warning_refs': ['warning_id', 'ref_type', 'ref_id', 'ref_url'],
}
"""Reads collected into published arrays, ordered by their whole projection."""

OUTPUT_COLUMNS = (
    'id',
    'clinicalStage',
    'origin',
    'type',
    'source',
    'provider',
    'drugs',
    'countries',
    'trialDescription',
    'phaseFromSource',
    'trialOfficialTitle',
    'trialSponsor',
    'diseases',
    'trialLiterature',
    'sideEffects',
    'trialPhase',
    'trialWhyStopped',
    'trialStudyType',
    'qualityControls',
    'trialPrimaryPurpose',
    'title',
    'trialNumberOfArms',
    'trialOverallStatus',
    'year',
    'trialStopReasonCategories',
    'url',
    'trialStartDate',
)
"""The columns of `output/clinical_report`, in their published order."""


class ClinicalReportFlags(StrEnum):
    PHASE_IV_NOT_APPROVED = 'PHASE_IV_NOT_APPROVED'
    UNVALIDATED_INDICATION = 'UNVALIDATED_INDICATION'
    INDIRECT_PRIMARY_PURPOSE = 'INDIRECT_PRIMARY_PURPOSE'
    NO_DISEASE = 'NO_DISEASE'
    NO_DRUGS = 'NO_DRUGS'
    SAFETY_REPORT = 'SAFETY_REPORT'


def clinical_report(
    source: dict[str, str],
    destination: dict[str, str],
    settings: dict[str, Any],
    config: Config,
) -> None:
    """Generate clinical report dataset from ChEMBL molecules and disease data.

    Args:
        source: Dictionary containing paths to input data
        destination: Dictionary containing paths to output data:
            - output: Path to write clinical reports (output/clinical_report)
        settings: Dictionary containing step variables, none used
        config: Config object, for the scratch space of the dump restores
    """
    logger.info(f'source paths: {source}')
    llm_extractions = EXTRACTION.read(source['trial_extraction'])
    stop_reasons = STOP_REASONS.read({'aact_stop_reasons': source['aact_stop_reasons']})
    drug_entities = DRUG_ENTITIES.read(
        {
            'aact_drug_entities': source['aact_drug_entities'],
            'mining_drug_entities': source['mining_drug_entities'],
        },
        expect={'aact_drug_entities': {'extraction_tag': EXTRACTION.release_tag()}},
    )
    disease_entities = DISEASE_ENTITIES.read({'mining_disease_entities': source['mining_disease_entities']})
    chembl_curation = scan_dataset(source['chembl_curation']).collect() if 'chembl_curation' in source else None

    logger.info(f'restoring chembl tables from {source["chembl"]}')
    chembl_tables = read_dump_tables(
        str(source['chembl']),
        CHEMBL_TABLES,
        schema_name=CHEMBL_SCHEMA_NAME,
        order_by=CHEMBL_ORDER_BY,
        scratch_root=config.work_path,
    )
    chembl_indication = chembl_tables['drug_indication']
    chembl_indication_references = chembl_tables['indication_refs']
    chembl_molecule_dictionary = chembl_tables['molecule_dictionary']
    chembl_drug_warning = chembl_tables['drug_warning']
    chembl_drug_warning_references = chembl_tables['warning_refs']

    logger.info(f'restoring aact tables from {source["aact"]}')
    aact_tables = read_aact_tables(
        str(source['aact']),
        ['studies', 'interventions', 'conditions', 'study_references', 'designs', 'brief_summaries', 'sponsors'],
        config.work_path,
    )
    aact_studies = aact_tables['studies']
    aact_interventions = aact_tables['interventions']
    aact_conditions = aact_tables['conditions']
    aact_study_references = aact_tables['study_references']
    aact_designs = aact_tables['designs']
    aact_summaries = aact_tables['brief_summaries']
    aact_lead_sponsors = aact_tables['sponsors'].filter(pl.col('lead_or_collaborator') == 'lead')

    molecule_index = scan_dataset(source['chembl_molecule']).select(DRUG_COLUMNS).collect()
    disease_index = scan_dataset(source['disease']).select(DISEASE_COLUMNS).collect()

    logger.info('extract clinical report')
    mining = staged_mining_reports(source, disease_entities)
    aact = extract_aact_clinical_report(
        studies=aact_studies,
        interventions=aact_interventions,
        conditions=aact_conditions,
        additional_metadata=[
            aact_study_references,
            aact_designs,
            aact_summaries,
            aact_lead_sponsors,
        ],
        aggregation_specs={
            'study_references': {
                'group_by': 'nct_id',
                'alias': 'literature',
                'struct': {'id': 'pmid', 'type': 'reference_type'},
                'agg': 'unique',
            },
            'sponsor': {
                'group_by': 'nct_id',
                'alias': 'sponsor',
                'struct': {
                    'agencyClass': 'agency_class',
                    'name': 'name',
                },
                'agg': 'first',
            },
        },
        llm_extractions=llm_extractions,
    )
    logger.info(f'attach stop reasons staged at {source["aact_stop_reasons"]}')
    aact = ClinicalReport(df=attach_stop_reasons(aact.df, stop_reasons))
    chembl_indication_report = extract_chembl_clinical_report(
        drug_indication=chembl_indication,
        molecule_dictionary=chembl_molecule_dictionary,
        indication_refs=chembl_indication_references,
    )
    chembl_drug_warning_report = extract_drug_warning_clinical_report(
        drug_warning=chembl_drug_warning,
        molecule_dictionary=chembl_molecule_dictionary,
        warning_refs=chembl_drug_warning_references,
    )

    reports = union_dfs([
        mining['pmda'].df,
        aact.df,
        # Remove ClinicalTrials reports from ChEMBL indications as they are already included in AACT
        chembl_indication_report.pipe(lambda self: ClinicalReport(filter_df(self.df, "source != 'ClinicalTrials'"))).df,
        chembl_drug_warning_report.df,
        mining['ttd'].df,
        mining['ema'].df,
    ])
    logger.info('reports generated. map entities...')
    output = (
        map_report_entities(
            reports=reports,
            disease_index=disease_index,
            drug_index=molecule_index,
            chembl_curation=chembl_curation,
            drug_entities=drug_entities,
        )
        .pipe(validate_disease, disease_index=disease_index.select('id', 'obsoleteTerms'))
        .pipe(create_title)
        .pipe(flag_null_diseases)
        .pipe(flag_null_drugs)
        .pipe(flag_safety_reports)
        .pipe(flag_phase_iv_not_approved)
        .pipe(flag_indirect_primary_purpose, llm_drug_intent=llm_extractions.select('id', 'drug_intent'))
        .pipe(flag_unvalidated_indication, chembl_indication_report=chembl_indication_report)
    )

    logger.info(f'destination paths: {destination}')
    write_dataset(order_output_columns(output.df), destination['output'])


def staged_mining_reports(source: dict[str, str], staged: pl.DataFrame) -> dict[str, ClinicalReport]:
    """The TTD, EMA and PMDA reports, with the EMA and PMDA diseases staged in `staged`.

    Args:
        source: the paths of the TTD, EMA and PMDA inputs.
        staged: the staged disease entities, with `text` and `extracted_diseases`.

    Raises:
        ValueError: if an EMA or PMDA indication text is not staged.
    """
    return mining_reports(
        read_mining_inputs(source['ttd'], source['ema'], source['pmda']),
        staged_lookup(staged, 'text', 'extracted_diseases', 'disease texts'),
    )


def attach_stop_reasons(aact: pl.DataFrame, staged: pl.DataFrame) -> pl.DataFrame:
    """The AACT reports with `trialStopReasonCategories`, the staged categories of their `trialWhyStopped`.

    Args:
        aact: AACT reports.
        staged: the staged stop reasons, with `text` and `categories`.

    Returns:
        `aact` with `trialStopReasonCategories`, null where `trialWhyStopped` is null.

    Raises:
        ValueError: if a `trialWhyStopped` is not staged.
    """
    require_staged(aact['trialWhyStopped'], staged['text'], 'stop reasons')
    return aact.join(
        staged.select(pl.col('text').alias('trialWhyStopped'), pl.col('categories').alias('trialStopReasonCategories')),
        on='trialWhyStopped',
        how='left',
        maintain_order='left',
    )


def map_report_entities(
    reports: pl.DataFrame,
    disease_index: pl.DataFrame,
    drug_index: pl.DataFrame,
    chembl_curation: pl.DataFrame | None,
    drug_entities: pl.DataFrame,
) -> ClinicalReport:
    """Map the drugs and diseases of each report to ids.

    Args:
        reports: the reports of every provider.
        disease_index: disease index, with `DISEASE_COLUMNS`.
        drug_index: drug molecules, with `DRUG_COLUMNS`.
        chembl_curation: ChEMBL's clinical trial curation.
        drug_entities: the staged drug NER, used in place of NER.

    Returns:
        ClinicalReport with the ids found.

    Raises:
        ValueError: if a drug label reaching NER is not staged.
    """
    exploded_reports = (
        reports
        .explode('drugs', empty_as_null=True)
        .explode('diseases', empty_as_null=True)
        .unnest(['drugs', 'diseases'])
    )
    mapped_exploded_reports = map_entities(exploded_reports, disease_index, drug_index, chembl_curation, drug_entities)

    mapped_reports = (
        mapped_exploded_reports
        .with_columns(
            disease=pl.when(
                pl.any_horizontal(pl.col('diseaseFromSource').is_not_null(), pl.col('diseaseId').is_not_null())
            ).then(pl.struct(pl.col('diseaseFromSource'), pl.col('diseaseId'))),
            drug=pl.when(
                pl.any_horizontal(pl.col('drugFromSource').is_not_null(), pl.col('drugId').is_not_null())
            ).then(pl.struct(pl.col('drugFromSource'), pl.col('drugId'))),
        )
        .drop(['diseaseFromSource', 'drugFromSource', 'diseaseId', 'drugId'])
        .unique()
    )

    return ClinicalReport(
        df=(
            mapped_reports
            .group_by([c for c in mapped_reports.columns if c not in ['disease', 'drug']])
            .agg(
                pl.col('disease').drop_nulls().unique().alias('diseases'),
                pl.col('drug').drop_nulls().unique().alias('drugs'),
            )
            .with_columns(pl.when(pl.col(c).list.len() > 0).then(pl.col(c)) for c in ['diseases', 'drugs'])
        )
    )


def order_output_columns(reports: pl.DataFrame) -> pl.DataFrame:
    """The reports' columns in the published order of `output/clinical_report`.

    Raises:
        ValueError: if the reports have a column `output/clinical_report` does not.
    """
    extra = [c for c in reports.columns if c not in OUTPUT_COLUMNS]
    if extra:
        raise ValueError(f'columns not in output/clinical_report: {extra}')
    return reports.select(OUTPUT_COLUMNS)


def validate_disease(reports: ClinicalReport, disease_index: pl.DataFrame) -> ClinicalReport:
    """Validate disease entities in the reports.

    Args:
        reports: ClinicalReport object with mapped entities
        disease_index: disease index, projected to the only columns used here: `id` and
            `obsoleteTerms`. The caller pushes that projection into the read.

    Returns:
        ClinicalReport object with validated disease entities
    """
    exploded = reports.df.explode('diseases', empty_as_null=True).unnest('diseases')
    diseases = disease_index.select(pl.col('id').alias('diseaseId'), 'obsoleteTerms').explode(
        'obsoleteTerms', empty_as_null=True
    )

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

    return ClinicalReport(
        df=(
            pl
            # combine all report dfs
            .concat([null_ids, valid_ids, updated_ids])
            # flag rows where disease info is null to avoid lists of empty elements
            .with_columns(noDiseaseInfo=pl.col('diseaseFromSource').is_null())
            .group_by(reports.df.drop('diseases').columns, maintain_order=True)
            .agg(
                pl.struct(['diseaseFromSource', 'diseaseId']).unique().alias('diseases'),
                pl.col('noDiseaseInfo').any().alias('noDiseaseInfo'),
            )
            .with_columns(diseases=pl.when(pl.col('noDiseaseInfo')).then(pl.lit(None)).otherwise(pl.col('diseases')))
            .drop('noDiseaseInfo')
        )
    )


def create_title(reports: ClinicalReport) -> ClinicalReport:
    """Create a title for the report based on the information on drugs and diseases."""
    return ClinicalReport(
        df=(
            reports.df
            .with_columns(
                drugs_count=pl.col('drugs').list.len(),
                # when diseases list is null, it is replaced with empty list
                diseases_count=pl.col('diseases').fill_null([]).list.len(),
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
                .when(pl.col('diseases_count') == 0)
                .then(pl.lit(None, dtype=pl.String))
                .when(pl.col('diseases_count') == 1)
                .then(pl.col('diseases').list.first().struct.field('diseaseFromSource').str.to_titlecase())
                .otherwise(pl.concat_str(pl.col('diseases_count').cast(pl.String), pl.lit(' diseases'))),
            )
            .with_columns(
                title=pl
                .when(pl.col('trialOfficialTitle').is_not_null())
                .then(pl.col('trialOfficialTitle'))
                .when(pl.col('diseases_count') == 0)
                .then(
                    pl.concat_str(
                        pl.lit('Report in '),
                        pl.col('_stage'),
                        pl.lit(' stage for '),
                        pl.col('_drug_part'),
                    )
                )
                .otherwise(
                    pl.concat_str(
                        pl.lit('Report in '),
                        pl.col('_stage'),
                        pl.lit(' stage for '),
                        pl.col('_drug_part'),
                        pl.lit(' and '),
                        pl.col('_disease_part'),
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
        .explode('drugs', empty_as_null=True)
        .explode('diseases', empty_as_null=True)
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


def flag_unvalidated_indication(
    reports: ClinicalReport,
    chembl_indication_report: ClinicalReport | None = None,
) -> ClinicalReport:
    """Adds flag to reports where any of the indications hasn't been seen in a simple study.

    Simple study: report with a single reported condition and drug
    """
    high_confidence_indications = pl.concat([
        # INDICATIONS FROM TRIALS REPORTING A SINGLE DISEASE AND DRUG
        (
            reports.df
            .explode('diseases', empty_as_null=True)
            .explode('drugs', empty_as_null=True)
            .unnest('drugs')
            .unnest('diseases')
            .group_by('id')
            .agg(
                pl.col('diseaseId').unique().alias('diseaseIds'),
                pl.col('drugId').unique().alias('drugIds'),
                pl.col('diseaseFromSource').unique().alias('diseaseFromSources'),
                pl.col('drugFromSource').unique().alias('drugFromSources'),
            )
            .filter((pl.col('diseaseFromSources').list.len() == 1) & (pl.col('drugFromSources').list.len() == 1))
            .select('diseaseIds', 'drugIds')
            .explode('diseaseIds', empty_as_null=True)
            .explode('drugIds', empty_as_null=True)
            .rename({'diseaseIds': 'diseaseId', 'drugIds': 'drugId'})
            .filter((pl.col('drugId').is_not_null()) & (pl.col('diseaseId').is_not_null()))
        ),
        # INDICATIONS FROM APPROVED/PREAPPROVED REPORTS
        (
            reports.df
            .filter(pl.col('clinicalStage').is_in([ClinicalStageCategory.APPROVAL, ClinicalStageCategory.PREAPPROVAL]))
            .explode('diseases', empty_as_null=True)
            .explode('drugs', empty_as_null=True)
            .select(pl.col('diseases').struct.field('diseaseId'), pl.col('drugs').struct.field('drugId'))
            .filter((pl.col('drugId').is_not_null()) & (pl.col('diseaseId').is_not_null()))
        ),
        # INDICATIONS FROM OFFICIAL CHEMBL RECORDS
        (
            chembl_indication_report.df
            .explode('diseases', empty_as_null=True)
            .explode('drugs', empty_as_null=True)
            .select(
                pl.col('diseases').struct.field('diseaseId'),
                pl.col('drugs').struct.field('drugId'),
            )
            .drop_nulls()
            .unique()
            if chembl_indication_report is not None
            else pl.DataFrame(schema={'diseaseId': pl.String, 'drugId': pl.String})
        ),
    ]).unique()

    # Flag reports where the drug/disease pair is NOT part of the high confidence set
    flagged_ids = (
        reports.df
        .explode('diseases', empty_as_null=True)
        .explode('drugs', empty_as_null=True)
        .select(pl.col('id'), pl.col('diseases').struct.field('diseaseId'), pl.col('drugs').struct.field('drugId'))
        .filter((pl.col('drugId').is_not_null()) & (pl.col('diseaseId').is_not_null()))
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


def flag_indirect_primary_purpose(
    reports: ClinicalReport, llm_drug_intent: pl.DataFrame | None = None
) -> ClinicalReport:
    """Flag reports where the primary purpose is not a therapeutic direct purpose.

    Args:
        reports: ClinicalReport to flag
        llm_drug_intent: DataFrame with drug_intent extracted from LLM

    Returns:
        ClinicalReport with indirect primary purpose flag
    """
    if llm_drug_intent is not None:
        reports_df = reports.df.join(llm_drug_intent, on='id', how='left')
        llm_drug_intent_condition = (
            # Flagging should only act on AACT reports
            (pl.col('source') == ClinicalSource.CLINICAL_TRIALS_GOV.value)
            &
            # Non-therapeutic intents (supportive_care, prevention, diagnostic, other)
            # or when LLM was not able to infer the intent (null)
            ((pl.col('drug_intent') != 'therapeutic') | pl.col('drug_intent').is_null())
        )
    else:
        reports_df = reports.df
        llm_drug_intent_condition = pl.lit(False)
    return ClinicalReport(
        df=update_quality_flag(
            df=reports_df,
            flag_condition=(
                # Flag consists of reports where the primary purpose is indirect
                pl.col('trialPrimaryPurpose').is_in(['DEVICE_FEASIBILITY', 'DIAGNOSTIC'])
                # or the extracted drug_intent is missing or non-therapeutic
                | llm_drug_intent_condition
            ),
            flag_text=ClinicalReportFlags.INDIRECT_PRIMARY_PURPOSE.value,
        ).select(pl.all().exclude('drug_intent'))
    )


def flag_null_diseases(reports: ClinicalReport) -> ClinicalReport:
    """Flag reports where the diseases column is null."""
    return ClinicalReport(
        df=update_quality_flag(
            df=reports.df,
            flag_condition=pl.col('diseases').is_null(),
            flag_text=ClinicalReportFlags.NO_DISEASE.value,
        )
    )


def flag_null_drugs(reports: ClinicalReport) -> ClinicalReport:
    """Flag reports where the drugs column is null."""
    return ClinicalReport(
        df=update_quality_flag(
            df=reports.df,
            flag_condition=pl.col('drugs').is_null(),
            flag_text=ClinicalReportFlags.NO_DRUGS.value,
        )
    )


def flag_safety_reports(reports: ClinicalReport) -> ClinicalReport:
    """Flag SAFETY-type reports (mira#57, informational, not a failure)."""
    return ClinicalReport(
        df=update_quality_flag(
            df=reports.df,
            flag_condition=pl.col('type').eq(ClinicalReportType.SAFETY.value),
            flag_text=ClinicalReportFlags.SAFETY_REPORT.value,
        )
    )
