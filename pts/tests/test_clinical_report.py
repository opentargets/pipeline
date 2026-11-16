import datetime
import json
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

import openpyxl
import polars as pl
import pytest
from otter.config.model import Config

from pts.transformers import clinical_report as clinical_report_module
from pts.transformers.clinical_mira import (
    ClinicalProvider,
    ClinicalReport,
    ClinicalReportType,
    ClinicalSource,
    extract_aact_clinical_report,
)
from pts.transformers.clinical_mira.mining import read_ema_sheet
from pts.transformers.clinical_mira.providers import aact as aact_provider
from pts.transformers.clinical_mira.providers.aact import replace_with_llm_indications
from pts.transformers.clinical_mira.providers.drugs_at_fda import extract_clinical_report as extract_fda_clinical_report
from pts.transformers.clinical_mira.providers.ema import extract_clinical_report as extract_ema_clinical_report
from pts.transformers.clinical_mira.record_store import publish_snapshot
from pts.transformers.clinical_mira.schemas import ClinicalReportExtractionSchema, ClinicalReportOrigin
from pts.transformers.clinical_mira.staged_outputs import (
    DISEASE_ENTITIES,
    DRUG_ENTITIES,
    EXTRACTION,
    EXTRACTION_V2,
    STOP_REASONS,
    StagedOutput,
    schema_digest,
)
from pts.transformers.clinical_mira.trial_links import report_entities_frame
from pts.transformers.clinical_report import (
    ClinicalReportFlags,
    attach_stop_reasons,
    flag_indirect_primary_purpose,
    flag_null_diseases,
    flag_observational_studies,
    map_broader_diseases,
    map_report_entities,
    order_output_columns,
    require_staged,
    trial_broader_terms,
    trial_drug_intents,
    validate_disease,
)
from tests.clinical_mira.extraction_v2_fixtures import v2_frame, v2_link
from tests.clinical_mira.mapping_fixtures import DISEASE, DRUG


def _build_reports(entries) -> ClinicalReport:
    return ClinicalReport(df=pl.DataFrame(entries))


def _disease_index() -> pl.DataFrame:
    return pl.DataFrame({
        'id': pl.Series('id', ['EFO:0009880'], dtype=pl.Utf8),
        'obsoleteTerms': pl.Series('obsoleteTerms', [[None]], dtype=pl.List(pl.Utf8)),
    })


def _report_entry(report_id: str, disease_struct: dict[str, str | None]) -> dict:
    return {
        'id': report_id,
        'phaseFromSource': 'black box warning',
        'type': ClinicalReportType.INDICATION.value,
        'origin': ClinicalReportOrigin.DRUG_LABEL.value,
        'source': ClinicalSource.DailyMed.value,
        'provider': ClinicalProvider.CHEMBL.value,
        'year': None,
        'countries': ['United States'],
        'url': 'https://example.org',
        'drugs': [{'drugFromSource': 'BENAZEPRIL', 'drugId': 'CHEMBL1694'}],
        'diseases': [disease_struct],
    }


def test_validate_disease_null_if_only_null_structs() -> None:
    """Test that validate_disease returns None when all disease structs are null."""
    reports = _build_reports([
        _report_entry('null_disease', {'diseaseFromSource': None, 'diseaseId': None}),
        _report_entry(
            'with_disease',
            {'diseaseFromSource': 'teratogenicity', 'diseaseId': 'EFO:0009880'},
        ),
    ])

    validated = validate_disease(reports, disease_index=_disease_index())
    diseases = validated.df.filter(pl.col('id') == 'null_disease').select('diseases').to_series().to_list()[0]

    assert diseases is None


def test_validate_disease_preserves_populated_diseases() -> None:
    """Test that validate_disease preserves populated disease structs."""
    reports = _build_reports([
        _report_entry(
            'with_disease',
            {'diseaseFromSource': 'teratogenicity', 'diseaseId': 'EFO:0009880'},
        ),
    ])

    validated = validate_disease(reports, disease_index=_disease_index())
    diseases = validated.df.filter(pl.col('id') == 'with_disease').select('diseases').to_series().to_list()[0]

    assert diseases == [
        {'diseaseFromSource': 'teratogenicity', 'diseaseId': 'EFO:0009880'},
    ]


def _report(id_: str, source: str, primary_purpose: str | None = None) -> dict:
    return {
        'id': id_,
        'phaseFromSource': 'phase 3',
        'type': ClinicalReportType.INDICATION.value,
        'source': source,
        'provider': (
            ClinicalProvider.AACT.value
            if source == ClinicalSource.CLINICAL_TRIALS_GOV.value
            else ClinicalProvider.CHEMBL.value
        ),
        'origin': (
            ClinicalReportOrigin.CLINICAL_TRIAL.value
            if source == ClinicalSource.CLINICAL_TRIALS_GOV.value
            else ClinicalReportOrigin.DRUG_LABEL.value
        ),
        'trialPrimaryPurpose': primary_purpose,
        'drugs': [{'drugFromSource': 'BENAZEPRIL', 'drugId': 'CHEMBL1694'}],
        'diseases': [{'diseaseFromSource': 'hypertension', 'diseaseId': 'EFO:0000537'}],
    }


def _flagged(reports: ClinicalReport, report_id: str) -> bool:
    qc = reports.df.filter(pl.col('id') == report_id).select('qualityControls').to_series().to_list()[0]
    return qc is not None and ClinicalReportFlags.INDIRECT_PRIMARY_PURPOSE.value in qc


def test_flag_observational_studies() -> None:
    entries = [
        _report('obs', ClinicalSource.CLINICAL_TRIALS_GOV.value) | {'trialStudyType': 'OBSERVATIONAL'},
        _report('int', ClinicalSource.CLINICAL_TRIALS_GOV.value) | {'trialStudyType': 'INTERVENTIONAL'},
        _report('ea', ClinicalSource.CLINICAL_TRIALS_GOV.value) | {'trialStudyType': 'EXPANDED_ACCESS'},
        _report('none', ClinicalSource.CLINICAL_TRIALS_GOV.value) | {'trialStudyType': None},
        _report('label', ClinicalSource.DailyMed.value) | {'trialStudyType': None},
    ]
    result = flag_observational_studies(_build_reports(entries)).df
    flagged = {
        row['id']
        for row in result.iter_rows(named=True)
        if ClinicalReportFlags.OBSERVATIONAL_STUDY.value in (row['qualityControls'] or [])
    }
    assert flagged == {'obs'}


def test_flag_indirect_primary_purpose_device_feasibility() -> None:
    reports = _build_reports([
        _report('r1', ClinicalSource.CLINICAL_TRIALS_GOV.value, primary_purpose='TREATMENT'),
        _report('r2', ClinicalSource.CLINICAL_TRIALS_GOV.value, primary_purpose='DEVICE_FEASIBILITY'),
        _report('r3', ClinicalSource.CLINICAL_TRIALS_GOV.value, primary_purpose='DIAGNOSTIC'),
        _report('r4', ClinicalSource.CLINICAL_TRIALS_GOV.value, primary_purpose='OTHER'),
    ])
    result = flag_indirect_primary_purpose(reports)
    assert not _flagged(result, 'r1')
    assert _flagged(result, 'r2')
    assert _flagged(result, 'r3')
    assert not _flagged(result, 'r4')


def test_flag_indirect_primary_purpose_no_primary_purpose() -> None:
    """Reports without a trialPrimaryPurpose should not be flagged."""
    reports = _build_reports([
        _report('r1', ClinicalSource.CLINICAL_TRIALS_GOV.value, primary_purpose=None),
    ])
    result = flag_indirect_primary_purpose(reports)
    assert not _flagged(result, 'r1')


def test_flag_indirect_primary_purpose_with_llm_no_match_is_flagged() -> None:
    """When llm_batch_results is provided but report has no match, drug_intent is null → flagged."""
    reports = _build_reports([
        _report('r1', ClinicalSource.CLINICAL_TRIALS_GOV.value, primary_purpose='TREATMENT'),
    ])
    llm_batch_results = pl.DataFrame({
        'id': ['r_other'],
        'drug_intent': ['therapeutic'],
    })
    result = flag_indirect_primary_purpose(reports, llm_drug_intent=llm_batch_results)
    assert _flagged(result, 'r1')


def test_flag_indirect_primary_purpose_with_llm_therapeutic_not_flagged() -> None:
    """drug_intent='therapeutic' should not be flagged."""
    reports = _build_reports([
        _report('r1', ClinicalSource.CLINICAL_TRIALS_GOV.value, primary_purpose='TREATMENT'),
    ])
    llm_batch_results = pl.DataFrame({'id': ['r1'], 'drug_intent': ['therapeutic']})
    result = flag_indirect_primary_purpose(reports, llm_drug_intent=llm_batch_results)
    assert not _flagged(result, 'r1')


def test_flag_indirect_primary_purpose_with_llm_non_therapeutic_flagged() -> None:
    """Non-therapeutic drug_intent values (prevention, supportive_care, etc.) should be flagged."""
    reports = _build_reports([
        _report('r1', ClinicalSource.CLINICAL_TRIALS_GOV.value, primary_purpose='TREATMENT'),
        _report('r2', ClinicalSource.CLINICAL_TRIALS_GOV.value, primary_purpose='TREATMENT'),
    ])
    llm_batch_results = pl.DataFrame({
        'id': ['r1', 'r2'],
        'drug_intent': ['therapeutic', 'supportive_care'],
    })
    result = flag_indirect_primary_purpose(reports, llm_drug_intent=llm_batch_results)
    assert not _flagged(result, 'r1')
    assert _flagged(result, 'r2')


def test_trial_drug_intents_match_reports_whose_ids_clinical_report_lowercased() -> None:
    """A v2 extraction keyed by the AACT NCT id reaches the lowercased AACT reports: a therapeutic trial is kept."""
    extractions = v2_frame({
        'NCT0001': {'drug_intent': 'therapeutic'},
        'NCT0002': {'drug_intent': 'supportive_care'},
    })
    reports = _build_reports([
        _report('NCT0001', ClinicalSource.CLINICAL_TRIALS_GOV.value, primary_purpose='TREATMENT'),
        _report('NCT0002', ClinicalSource.CLINICAL_TRIALS_GOV.value, primary_purpose='TREATMENT'),
    ])

    intents = trial_drug_intents(extractions)
    result = flag_indirect_primary_purpose(reports, llm_drug_intent=intents)

    assert intents.columns == ['id', 'drug_intent']
    assert sorted(intents['id'].to_list()) == ['nct0001', 'nct0002']
    assert not _flagged(result, 'nct0001')
    assert _flagged(result, 'nct0002')


BENAZEPRIL = {'drugFromSource': 'BENAZEPRIL', 'drugId': 'CHEMBL1694'}


def _prevention_trial(id_: str, **changes: Any) -> dict:
    return _report(id_, ClinicalSource.CLINICAL_TRIALS_GOV.value, primary_purpose='PREVENTION') | changes


def _flag_as_prevention(entries: list[dict]) -> ClinicalReport:
    intents = pl.DataFrame({'id': [e['id'] for e in entries], 'drug_intent': ['prevention'] * len(entries)})
    return flag_indirect_primary_purpose(_build_reports(entries), llm_drug_intent=intents)


def test_prevention_trial_of_one_drug_is_not_flagged() -> None:
    result = _flag_as_prevention([
        _prevention_trial('prevention'),
        _prevention_trial('treatment', trialPrimaryPurpose='TREATMENT'),
        _prevention_trial(
            'same drug twice',
            drugs=[
                {'drugFromSource': 'BENAZEPRIL', 'drugId': 'CHEMBL1694'},
                {'drugFromSource': 'benazepril hydrochloride', 'drugId': 'CHEMBL1694'},
            ],
        ),
    ])
    assert not any(_flagged(result, id_) for id_ in ('prevention', 'treatment', 'same drug twice'))


@pytest.mark.parametrize(
    'changes',
    [
        {'drugs': [BENAZEPRIL, {'drugFromSource': 'X', 'drugId': 'CHEMBL2'}]},
        {'drugs': [BENAZEPRIL, {'drugFromSource': 'unmapped', 'drugId': None}]},
        {'phaseFromSource': 'NA'},
        {'trialPrimaryPurpose': 'BASIC_SCIENCE'},
        {'trialPrimaryPurpose': None},
        {'diseases': [{'diseaseFromSource': 'relapse', 'diseaseId': 'EFO_0004952'}]},
    ],
    ids=['two drugs', 'an unmapped second drug', 'no phase', 'basic science', 'no purpose', 'disease recurrence'],
)
def test_prevention_trial_stays_flagged(changes: dict) -> None:
    assert _flagged(_flag_as_prevention([_prevention_trial('r1', **changes)]), 'r1')


def test_flag_indirect_primary_purpose_drops_drug_intent() -> None:
    """The drug_intent column must be dropped from the output."""
    reports = _build_reports([
        _report('r1', ClinicalSource.CLINICAL_TRIALS_GOV.value, primary_purpose='TREATMENT'),
    ])
    llm_batch_results = pl.DataFrame({'id': ['r1'], 'drug_intent': ['therapeutic']})
    result = flag_indirect_primary_purpose(reports, llm_drug_intent=llm_batch_results)
    assert 'drug_intent' not in result.df.columns


def test_flag_indirect_primary_purpose_non_aact_not_flagged() -> None:
    """Non-AACT sources should not be flagged based on drug_intent, even if null or non-therapeutic."""
    reports = _build_reports([
        _report('r1', ClinicalSource.DailyMed.value, primary_purpose=None),
        _report('r2', ClinicalSource.DailyMed.value, primary_purpose=None),
        _report('r3', ClinicalSource.DailyMed.value, primary_purpose=None),
    ])
    llm_batch_results = pl.DataFrame({
        'id': ['r1', 'r2'],
        'drug_intent': ['therapeutic', 'prevention'],
    })
    result = flag_indirect_primary_purpose(reports, llm_drug_intent=llm_batch_results)
    assert not _flagged(result, 'r1')
    assert not _flagged(result, 'r2')
    assert not _flagged(result, 'r3')


def _extraction(nct_id: str, disease: str, drug: str, confidence: float = 0.9) -> pl.DataFrame:
    """A one-link frame in `TrialExtractionV2` shape, as the extraction snapshot holds it."""
    return v2_frame({nct_id: {'links': [v2_link(drug, disease, confidence)]}})


def test_registry_entities_are_replaced_by_the_extraction() -> None:
    studies = pl.DataFrame({
        'nct_id': ['NCT0001', 'NCT0001', 'NCT0002'],
        'diseaseFromSource': ['from-aact', 'pain', 'from-aact'],
        'drugFromSource': ['from-aact', 'from-aact', 'from-aact'],
    })
    entities = report_entities_frame(_extraction('nct0001', 'asthma', 'aspirin'), 0.8)

    result = replace_with_llm_indications(studies, entities)

    assert result.rows() == [('NCT0001', 'asthma', 'aspirin')]


def _aact_tables(nct_id: str = 'NCT0001') -> dict[str, pl.DataFrame]:
    """One interventional drug trial, as the AACT tables give it."""
    return {
        'studies': pl.DataFrame({
            'nct_id': [nct_id],
            'study_type': ['INTERVENTIONAL'],
            'phase': ['PHASE2'],
            'official_title': ['t'],
        }),
        'interventions': pl.DataFrame({'nct_id': [nct_id], 'intervention_type': ['DRUG'], 'name': ['Registry Drug']}),
        'conditions': pl.DataFrame({'nct_id': [nct_id], 'downcase_name': ['registry condition']}),
    }


def _entities(row: dict) -> tuple[pl.DataFrame, pl.DataFrame]:
    """The extraction of NCT0001 with `row`, and its report entities at confidence 0.8."""
    extraction = v2_frame({'NCT0001': row})
    return extraction, report_entities_frame(extraction, 0.8)


def test_trial_report_entities_come_from_v2_links_above_threshold() -> None:
    _, entities = _entities({
        'links': [v2_link('aspirin', 'asthma', 0.9), v2_link('ibuprofen', 'pain', 0.5)],
        'background_drugs': [{'drug': 'cisplatin', 'synonyms': [], 'role': 'backbone', 'evidence_quote': 'q'}],
        'population_conditions': ['hypertension'],
    })

    (row,) = extract_aact_clinical_report(**_aact_tables(), trial_entities=entities).df.iter_rows(named=True)

    assert row['drugs'] == [{'drugFromSource': 'aspirin', 'drugId': None}]
    assert row['diseases'] == [{'diseaseId': None, 'diseaseFromSource': 'asthma'}]


def _mapped(entities: pl.DataFrame) -> ClinicalReport:
    reports = extract_aact_clinical_report(**_aact_tables(), trial_entities=entities).df
    return map_report_entities(reports, DISEASE, DRUG, chembl_curation=None, drug_entities=_staged_drug_entities())


def test_unmapped_indication_falls_back_to_its_broader_term() -> None:
    _, entities = _entities({
        'links': [
            v2_link('aspirin', 'cystic fibrosis in adults', broader='cystic fibrosis'),
            v2_link('aspirin', 'made-up disease', broader='made-up family'),
        ]
    })
    mapped = _mapped(entities)

    result = map_broader_diseases(
        mapped,
        trial_broader_terms(entities),
        DISEASE,
        DRUG,
        chembl_curation=None,
        drug_entities=_staged_drug_entities(),
    )

    (row,) = result.df.iter_rows(named=True)
    assert sorted(row['diseases'], key=str) == sorted(
        [
            {'diseaseFromSource': 'cystic fibrosis in adults', 'diseaseId': 'MONDO_0009061'},
            {'diseaseFromSource': 'made-up disease', 'diseaseId': None},
        ],
        key=str,
    )
    assert row['drugs'] == [{'drugFromSource': 'aspirin', 'drugId': 'CHEMBL25'}]
    assert result.df.drop('diseases').equals(mapped.df.drop('diseases'))


def test_broader_terms_map_only_trial_reports() -> None:
    _, entities = _entities({'links': [v2_link('aspirin', 'cf in adults', broader='cystic fibrosis')]})
    other = {
        **_report('NCT0001', ClinicalSource.TTD.value),
        'diseases': [{'diseaseFromSource': 'cf in adults', 'diseaseId': None}],
    }
    reports = ClinicalReport(df=pl.DataFrame([other]))

    result = map_broader_diseases(
        reports,
        trial_broader_terms(entities),
        DISEASE,
        DRUG,
        chembl_curation=None,
        drug_entities=_staged_drug_entities(),
    )

    assert result.df['diseases'].to_list() == [[{'diseaseFromSource': 'cf in adults', 'diseaseId': None}]]


def test_trial_without_confident_links_has_no_report() -> None:
    _, entities = _entities({'links': [v2_link('aspirin', 'cystic fibrosis', 0.5)]})

    reports = extract_aact_clinical_report(**_aact_tables('NCT0001'), trial_entities=entities).df

    assert reports.is_empty()


def test_trial_the_extraction_misses_has_no_report() -> None:
    _, entities = _entities({'links': [v2_link('aspirin', 'asthma')]})

    reports = extract_aact_clinical_report(**_aact_tables('NCT0002'), trial_entities=entities).df

    assert reports.is_empty()


EMA_BANNER = [
    'Content type:',
    'Medicine',
    'Output automatically generated from content on www.ema.europa.eu on:',
    '27/05/2026 - 06:15',
]
EMA_HEADER = [
    'Category',
    'Name of medicine',
    'EMA product number',
    'Medicine status',
    'International non-proprietary name (INN) / common name',
    'Active substance',
    'Therapeutic area (MeSH)',
    'Therapeutic indication',
    'Marketing authorisation date',
    'Medicine URL',
    'Revision number',
]
EMA_ROWS = [
    [
        'Human',
        'Alpha',
        'EMEA/H/C/000001',
        'Authorised',
        'alphamab',
        'alphamab',
        'Asthma',
        '  treatment of asthma  ',
        '15/01/2020',
        'https://www.ema.europa.eu/en/medicines/human/EPAR/alpha',
        3,
    ],
    [
        'Human',
        'Beta',
        'EMEA/H/C/000002',
        'Withdrawn',
        'N/A',
        'betanib',
        None,
        'treatment of copd',
        datetime.datetime(2019, 3, 4),
        'https://www.ema.europa.eu/en/medicines/human/EPAR/beta',
        2.5,
    ],
    [
        'Human',
        'Gamma',
        'EMEA/H/C/000003',
        'Authorised',
        'gammastat',
        'gammastat',
        None,
        'nothing to extract',
        '01/02/2003',
        'https://www.ema.europa.eu/en/medicines/human/EPAR/gamma',
        None,
    ],
    [
        'Human',
        'Delta',
        'EMEA/H/C/000005',
        'Authorised',
        None,
        '   ',
        'None',
        'treatment of copd',
        '01/02/2003',
        'https://www.ema.europa.eu/en/medicines/human/EPAR/delta',
        ' \t ',
    ],
    [
        'Veterinary',
        'Vetix',
        'EMEA/V/C/000004',
        'Authorised',
        None,
        'vetixin',
        None,
        'mastitis',
        '01/02/2003',
        'https://www.ema.europa.eu/en/medicines/veterinary/EPAR/vetix',
        1,
    ],
]

EMA_NER = {
    'treatment of asthma': ['asthma'],
    'treatment of copd': ['chronic obstructive pulmonary disease'],
    'nothing to extract': [],
}

READ_EXCEL_COLUMNS = [*EMA_BANNER, *(f'__UNNAMED__{i}' for i in range(len(EMA_BANNER), len(EMA_HEADER)))]
READ_EXCEL_ROWS = [
    tuple(EMA_HEADER),
    (
        'Human',
        'Alpha',
        'EMEA/H/C/000001',
        'Authorised',
        'alphamab',
        'alphamab',
        'Asthma',
        '  treatment of asthma  ',
        '15/01/2020',
        'https://www.ema.europa.eu/en/medicines/human/EPAR/alpha',
        '3',
    ),
    (
        'Human',
        'Beta',
        'EMEA/H/C/000002',
        'Withdrawn',
        None,
        'betanib',
        None,
        'treatment of copd',
        '2019-03-04 00:00:00',
        'https://www.ema.europa.eu/en/medicines/human/EPAR/beta',
        '2.5',
    ),
    (
        'Human',
        'Gamma',
        'EMEA/H/C/000003',
        'Authorised',
        'gammastat',
        'gammastat',
        None,
        'nothing to extract',
        '01/02/2003',
        'https://www.ema.europa.eu/en/medicines/human/EPAR/gamma',
        None,
    ),
    (
        'Human',
        'Delta',
        'EMEA/H/C/000005',
        'Authorised',
        None,
        '',
        None,
        'treatment of copd',
        '01/02/2003',
        'https://www.ema.europa.eu/en/medicines/human/EPAR/delta',
        '',
    ),
    (
        'Veterinary',
        'Vetix',
        'EMEA/V/C/000004',
        'Authorised',
        None,
        'vetixin',
        None,
        'mastitis',
        '01/02/2003',
        'https://www.ema.europa.eu/en/medicines/veterinary/EPAR/vetix',
        '1',
    ),
]
"""`pl.read_excel(<bytes>, sheet_name='Medicine')` of `_ema_workbook`, captured with fastexcel 0.21.0.

Its 'N/A' and 'None' cells are set to null, as `read_ema_sheet` reads them.
"""


def _ema_workbook(path: Path, blank_rows: int = 0) -> str:
    """The EMA medicines workbook: a banner row, `blank_rows` blank rows, the header, then the medicines."""
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = 'Medicine'
    for i, row in enumerate([EMA_BANNER, *[[None]] * blank_rows, EMA_HEADER, *EMA_ROWS]):
        for j, value in enumerate(row):
            sheet.cell(row=i + 1, column=j + 1, value=value)
    workbook.save(path)
    return str(path)


def _stub_ner(texts: pl.Series) -> pl.Series:
    return pl.Series([EMA_NER[text] for text in texts], dtype=pl.List(pl.String))


def _sorted_reports(reports: ClinicalReport) -> list[dict]:
    return reports.df.sort('id').with_columns(pl.col('drugs', 'diseases').list.sort()).to_dicts()


def test_ema_sheet_matches_read_excel(tmp_path: Path) -> None:
    """The sheet is `pl.read_excel`'s, except missing-value strings ('N/A', 'None', ...), null by design."""
    sheet = read_ema_sheet(_ema_workbook(tmp_path / 'ema.xlsx'))
    read_excel = pl.DataFrame(READ_EXCEL_ROWS, schema=dict.fromkeys(READ_EXCEL_COLUMNS, pl.String), orient='row')

    assert sheet.dtypes == [pl.String] * len(READ_EXCEL_COLUMNS)
    assert sheet.columns[: len(EMA_BANNER)] == EMA_BANNER
    assert sheet.rows() == READ_EXCEL_ROWS
    assert _sorted_reports(extract_ema_clinical_report(sheet, _stub_ner)) == _sorted_reports(
        extract_ema_clinical_report(read_excel, _stub_ner)
    )


def test_ema_sheet_skips_blank_rows_below_the_banner(tmp_path: Path) -> None:
    sheet = read_ema_sheet(_ema_workbook(tmp_path / 'ema.xlsx', blank_rows=7))

    assert sheet.columns == read_ema_sheet(_ema_workbook(tmp_path / 'plain.xlsx')).columns
    assert sheet.rows() == READ_EXCEL_ROWS
    assert sorted(extract_ema_clinical_report(sheet, _stub_ner).df['id'].to_list()) == [
        'emea/h/c/000001',
        'emea/h/c/000002',
        'emea/h/c/000005',
    ]


def test_ema_row_without_mesh_or_ner_dropped(tmp_path: Path) -> None:
    sheet = read_ema_sheet(_ema_workbook(tmp_path / 'ema.xlsx'))

    reports = extract_ema_clinical_report(sheet, _stub_ner).df

    assert sorted(reports['id'].to_list()) == ['emea/h/c/000001', 'emea/h/c/000002', 'emea/h/c/000005']


def test_ema_missing_value_inn_falls_back_to_the_active_substance(tmp_path: Path) -> None:
    sheet = read_ema_sheet(_ema_workbook(tmp_path / 'ema.xlsx'))

    reports = extract_ema_clinical_report(sheet, _stub_ner).df

    drugs = reports.filter(pl.col('id') == 'emea/h/c/000002')['drugs'].to_list()
    assert drugs == [[{'drugFromSource': 'betanib', 'drugId': None}]]


PUBLISHED_COLUMNS = [
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
]
"""The parquet column order of `gs://open-targets-pipeline-runs/do/platform-2610-2/output/clinical_report`."""


def test_output_column_order_matches_published() -> None:
    reports = pl.DataFrame({column: [None] for column in sorted(PUBLISHED_COLUMNS, reverse=True)})

    assert order_output_columns(reports).columns == PUBLISHED_COLUMNS


def test_output_column_order_rejects_unpublished_columns() -> None:
    reports = pl.DataFrame({column: [None] for column in [*PUBLISHED_COLUMNS, 'drug_intent']})

    with pytest.raises(ValueError, match='drug_intent'):
        order_output_columns(reports)


def test_map_report_entities_maps_each_drug_and_disease_of_a_report() -> None:
    reports = pl.DataFrame(
        [
            {
                **_report('NCT01', ClinicalSource.CLINICAL_TRIALS_GOV.value),
                'drugs': [
                    {'drugFromSource': 'Aspirin', 'drugId': None},
                    {'drugFromSource': 'Kalydeco', 'drugId': None},
                ],
                'diseases': [{'diseaseFromSource': 'Cystic fibrosis', 'diseaseId': None}],
            },
            {
                **_report('NCT02', ClinicalSource.CLINICAL_TRIALS_GOV.value),
                'drugs': [{'drugFromSource': 'Ecotrin', 'drugId': None}],
                'diseases': None,
            },
        ],
        schema_overrides={'diseases': pl.List(pl.Struct({'diseaseFromSource': pl.String, 'diseaseId': pl.String}))},
    )

    mapped = map_report_entities(
        reports,
        DISEASE,
        DRUG,
        chembl_curation=None,
        drug_entities=_staged_drug_entities(),
    )

    rows = {
        r['id']: (sorted(r['drugs'], key=str), r['diseases'])
        for r in mapped.df.select('id', 'drugs', 'diseases').iter_rows(named=True)
    }
    assert rows == {
        'nct01': (
            [
                {'drugFromSource': 'Aspirin', 'drugId': 'CHEMBL25'},
                {'drugFromSource': 'Kalydeco', 'drugId': 'CHEMBL2010601'},
            ],
            [{'diseaseFromSource': 'Cystic fibrosis', 'diseaseId': 'MONDO_0009061'}],
        ),
        'nct02': ([{'drugFromSource': 'Ecotrin', 'drugId': 'CHEMBL25'}], None),
    }


STAGED_DRUG_ENTITIES_SCHEMA = {'label': pl.String, 'extracted_drugs': pl.List(pl.String)}


def _staged_drug_entities(*rows: tuple[str, list[str]]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=STAGED_DRUG_ENTITIES_SCHEMA, orient='row')


def _staged_stop_reasons(*rows: tuple[str, list[str]]) -> pl.DataFrame:
    return pl.DataFrame(
        [(text, f'sha-{text}', categories) for text, categories in rows],
        schema={'text': pl.String, 'text_sha256': pl.String, 'categories': pl.List(pl.String)},
        orient='row',
    )


def _drug_reports(*rows: tuple[str, str, str]) -> pl.DataFrame:
    """Reports of (id, source, drug label), each with one unmapped drug and no disease."""
    return pl.DataFrame(
        [
            {
                **_report(id_, source),
                'drugs': [{'drugFromSource': label, 'drugId': None}],
                'diseases': None,
            }
            for id_, source, label in rows
        ],
        schema_overrides={'diseases': pl.List(pl.Struct({'diseaseFromSource': pl.String, 'diseaseId': pl.String}))},
    )


def _drug_ids(mapped: ClinicalReport) -> dict[tuple[str, str], list]:
    """The drug ids of each (source, drug label) of the mapped reports."""
    drugs = mapped.df.select('source', 'drugs').explode('drugs').unnest('drugs')
    return {
        (row['source'], row['drugFromSource']): sorted(row['drugId'], key=str)
        for row in drugs.group_by('source', 'drugFromSource').agg('drugId').iter_rows(named=True)
    }


def test_stop_reasons_come_from_staging() -> None:
    aact = pl.DataFrame({
        'id': ['NCT01', 'NCT02', 'NCT03', 'NCT04'],
        'trialWhyStopped': ['funding', None, '', 'funding'],
    })
    staged = _staged_stop_reasons(('funding', ['Insufficient_Funding']), ('', ['Uncategorised']), ('other', ['Other']))

    attached = attach_stop_reasons(aact, staged)

    assert attached.columns == ['id', 'trialWhyStopped', 'trialStopReasonCategories']
    assert attached.schema['trialStopReasonCategories'] == pl.List(pl.String)
    assert sorted(attached.rows()) == [
        ('NCT01', 'funding', ['Insufficient_Funding']),
        ('NCT02', None, None),
        ('NCT03', '', ['Uncategorised']),
        ('NCT04', 'funding', ['Insufficient_Funding']),
    ]


def test_missing_staged_why_stopped_fails_with_count_and_examples() -> None:
    missing = [f'reason {i}' for i in range(7)]
    aact = pl.DataFrame({'id': [f'NCT{i}' for i in range(9)], 'trialWhyStopped': [*missing, 'funding', 'funding']})
    staged = _staged_stop_reasons(('funding', ['Insufficient_Funding']))

    with pytest.raises(ValueError, match='7 stop reasons') as error:
        attach_stop_reasons(aact, staged)
    assert sum(text in str(error.value) for text in missing) == 5


def test_require_staged_ignores_nulls_and_repeats() -> None:
    require_staged(pl.Series(['a', None, 'a', '']), pl.Series(['', 'a']), 'labels')


def test_staged_names_serve_a_label_of_every_source() -> None:
    reports = _drug_reports(
        ('NCT01', ClinicalSource.CLINICAL_TRIALS_GOV.value, 'mystery pill'),
        ('TTD01', ClinicalSource.TTD.value, 'mystery pill'),
        ('X01', None, 'unknownium'),
    )

    mapped = map_report_entities(
        reports,
        DISEASE,
        DRUG,
        chembl_curation=None,
        drug_entities=_staged_drug_entities(('mystery pill', ['kalydeco']), ('unknownium', [])),
    )

    assert _drug_ids(mapped) == {
        (ClinicalSource.CLINICAL_TRIALS_GOV.value, 'mystery pill'): ['CHEMBL2010601'],
        (ClinicalSource.TTD.value, 'mystery pill'): ['CHEMBL2010601'],
        (None, 'unknownium'): [None],
    }


def test_unstaged_label_from_any_source_fails() -> None:
    reports = _drug_reports(
        ('NCT01', ClinicalSource.CLINICAL_TRIALS_GOV.value, 'mystery pill'),
        ('TTD01', ClinicalSource.TTD.value, 'ttd unstaged'),
        ('CHEMBL_IND1', ClinicalSource.DailyMed.value, 'chembl unstaged'),
        ('CHEMBL_IND2', ClinicalSource.DailyMed.value, 'Aspirin'),
    )

    with pytest.raises(ValueError, match=r"^2 drug labels not staged, e\.g\. \['chembl unstaged', 'ttd unstaged'\]$"):
        map_report_entities(
            reports,
            DISEASE,
            DRUG,
            chembl_curation=None,
            drug_entities=_staged_drug_entities(('mystery pill', ['kalydeco'])),
        )


def _snapshot_metadata(kind: StagedOutput, tag: str | None = None, **extra: str) -> dict[str, Any]:
    return {
        'kind': kind.name,
        'model': kind.model,
        'tag': tag or kind.tag(dict.fromkeys(kind.hub_models, 'r1')),
        'model_revisions': dict.fromkeys(kind.hub_models, 'r1'),
        **extra,
    }


def _release_source(
    tmp_path: Path,
    disease_tag: str | None = None,
    drug_extraction_tag: str | None = None,
    extraction_metadata: dict[str, Any] | None = None,
) -> dict[str, str]:
    """The sources of a release, staged under ``tmp_path``, each with the metadata given."""
    extraction_tag = drug_extraction_tag or EXTRACTION_V2.release_tag()
    metadatas = {
        'aact_stop_reasons': _snapshot_metadata(STOP_REASONS),
        'aact_drug_entities': _snapshot_metadata(DRUG_ENTITIES, extraction_tag=extraction_tag),
        'mining_drug_entities': _snapshot_metadata(DRUG_ENTITIES),
        'mining_disease_entities': _snapshot_metadata(DISEASE_ENTITIES, disease_tag),
    }
    kinds = {
        'aact_stop_reasons': STOP_REASONS,
        'aact_drug_entities': DRUG_ENTITIES,
        'mining_drug_entities': DRUG_ENTITIES,
        'mining_disease_entities': DISEASE_ENTITIES,
    }
    source = {'chembl': 'unused', 'aact': 'unused'}
    for name, metadata in metadatas.items():
        empty = pl.DataFrame(schema=dict.fromkeys(kinds[name].columns.values(), pl.String))
        publish_snapshot(empty, str(tmp_path / name), metadata, None)
        source[name] = str(tmp_path / name)
    extraction = _extraction('NCT0001', 'asthma', 'aspirin')
    metadata = extraction_metadata or {**EXTRACTION_V2.metadata('0' * 64, None), 'tag': EXTRACTION_V2.release_tag()}
    publish_snapshot(extraction, str(tmp_path / 'extraction'), metadata, None)
    source['trial_extraction'] = str(tmp_path / 'extraction')
    (tmp_path / 'stage_downgrades.tsv').write_text('ttdDrugId\tdrugFromSource\tdiseaseFromSource\treason\n')
    (tmp_path / 'link_corrections.tsv').write_text(
        'ttdDrugId\tdrugFromSource\tdiseaseFromSource\tdrugId\tdiseaseId\treason\n'
    )
    source['ttd_stage_downgrades'] = str(tmp_path / 'stage_downgrades.tsv')
    source['ttd_link_corrections'] = str(tmp_path / 'link_corrections.tsv')
    (tmp_path / 'indication_corrections.tsv').write_text(
        'reportId\tdrugFromSource\tdiseaseFromSource\tdiseaseId\tcorrectedDiseaseId\treason\n'
    )
    source['chembl_indication_corrections'] = str(tmp_path / 'indication_corrections.tsv')
    source['ema_indication_corrections'] = str(tmp_path / 'indication_corrections.tsv')
    with zipfile.ZipFile(tmp_path / 'drugs_at_fda.json.zip', 'w') as archive:
        archive.writestr('drug-drugsfda-0001-of-0001.json', json.dumps({'results': []}))
    source['drugs_at_fda'] = str(tmp_path / 'drugs_at_fda.json.zip')
    return source


@pytest.fixture
def no_dump(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError('dump restored')

    monkeypatch.setattr(clinical_report_module, 'read_dump_tables', refuse)
    monkeypatch.setattr(aact_provider, 'read_dump_tables', refuse)


def test_release_restores_todays_aact_tables(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    studies = [
        'nct_id',
        'overall_status',
        'phase',
        'study_type',
        'start_date',
        'why_stopped',
        'number_of_arms',
        'official_title',
    ]
    baseline = {
        'studies': studies,
        'interventions': ['nct_id', 'intervention_type', 'name'],
        'conditions': ['nct_id', 'downcase_name'],
        'study_references': ['nct_id', 'pmid', 'reference_type'],
        'designs': ['nct_id', 'primary_purpose'],
        'brief_summaries': ['nct_id', 'description'],
        'sponsors': ['nct_id', 'agency_class', 'lead_or_collaborator', 'name'],
    }
    captured: dict[str, Any] = {}

    class StopError(Exception):
        pass

    def aact_restore(source: str, tables: Any, **kwargs: Any) -> None:
        captured.update(tables=dict(tables), **kwargs)
        raise StopError

    chembl = dict.fromkeys(
        ['drug_indication', 'indication_refs', 'molecule_dictionary', 'drug_warning', 'warning_refs'],
        pl.DataFrame(),
    )
    monkeypatch.setattr(clinical_report_module, 'read_dump_tables', lambda *args, **kwargs: chembl)
    monkeypatch.setattr(aact_provider, 'read_dump_tables', aact_restore)

    with pytest.raises(StopError):
        clinical_report_module.clinical_report(
            _release_source(tmp_path),
            {},
            {},
            Config(step='clinical_report', steps=['clinical_report'], work_path=tmp_path),
        )

    assert captured['tables'] == baseline
    assert captured['order_by'] == {'study_references': ['nct_id', 'pmid', 'reference_type']}
    assert str(captured['scratch_root']) == str(tmp_path)


@pytest.mark.usefixtures('no_dump')
def test_release_fails_before_restoring_dumps_on_a_stale_snapshot(tmp_path: Path) -> None:
    source = _release_source(tmp_path, disease_tag='r1-pdeadbeef')
    with pytest.raises(ValueError, match=r"mining_disease_entities.*'deadbeef'"):
        clinical_report_module.clinical_report(source, {}, {}, None)


@pytest.mark.usefixtures('no_dump')
def test_release_fails_before_restoring_dumps_on_an_extraction_of_another_tag(tmp_path: Path) -> None:
    source = _release_source(tmp_path, extraction_metadata=EXTRACTION_V2.metadata('f' * 64, None))
    with pytest.raises(ValueError, match=r'ffffffffffff.*this code expects'):
        clinical_report_module.clinical_report(source, {}, {}, None)


@pytest.mark.usefixtures('no_dump')
def test_clinical_report_refuses_a_v1_extraction(tmp_path: Path) -> None:
    v1 = EXTRACTION.metadata(schema_digest(ClinicalReportExtractionSchema), None)
    source = _release_source(tmp_path, extraction_metadata=v1)
    with pytest.raises(ValueError, match=r"kind 'extraction', not 'extraction_v2'"):
        clinical_report_module.clinical_report(source, {}, {}, None)


@pytest.mark.usefixtures('no_dump')
def test_release_fails_before_restoring_dumps_on_an_aact_drug_snapshot_of_another_extraction(tmp_path: Path) -> None:
    source = _release_source(tmp_path, drug_extraction_tag='ffffffffffff-pdeadbeef')
    with pytest.raises(
        ValueError, match=r'aact_drug_entities extraction_tag expected .*found .*ffffffffffff-pdeadbeef'
    ):
        clinical_report_module.clinical_report(source, {}, {}, None)


@pytest.mark.usefixtures('no_dump')
def test_release_fails_before_restoring_dumps_on_an_invalid_ttd_curation(tmp_path: Path) -> None:
    source = _release_source(tmp_path)
    Path(source['ttd_stage_downgrades']).write_text(
        'ttdDrugId\tdrugFromSource\tdiseaseFromSource\treason\nD0B002\tb\tAsthma\ta\nD0B002\tb\tasthma\tb\n'
    )
    with pytest.raises(ValueError, match='duplicate'):
        clinical_report_module.clinical_report(source, {}, {}, None)


@pytest.mark.usefixtures('no_dump')
def test_release_fails_before_restoring_dumps_on_an_unreadable_drugs_at_fda_file(tmp_path: Path) -> None:
    source = _release_source(tmp_path)
    Path(source['drugs_at_fda']).write_text('not a zip')
    with pytest.raises(zipfile.BadZipFile):
        clinical_report_module.clinical_report(source, {}, {}, None)


@pytest.mark.usefixtures('no_dump')
def test_release_fails_on_an_aact_drug_snapshot_without_an_extraction_tag(tmp_path: Path) -> None:
    source = _release_source(tmp_path)
    metadata_path = Path(source['aact_drug_entities']) / 'metadata.json'
    metadata = json.loads(metadata_path.read_text())
    del metadata['extraction_tag']
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match=r'aact_drug_entities extraction_tag expected'):
        clinical_report_module.clinical_report(source, {}, {}, None)


RELEASE_MAPPING = """
import sys

import polars as pl

from pts.transformers.clinical_mira.mining import staged_lookup
import pts.transformers.clinical_mira.staged_outputs
from pts.transformers.clinical_report import map_report_entities
from tests.clinical_mira.mapping_fixtures import DISEASE, DRUG
from tests.test_clinical_report import _drug_reports, _staged_drug_entities

drugs = _staged_drug_entities(('mystery pill', ['kalydeco']))
map_report_entities(_drug_reports(('TTD01', 'TTD', 'mystery pill')), DISEASE, DRUG, None, drugs)
diseases = pl.DataFrame({'text': ['asthma'], 'extracted_diseases': [['asthma']]})
staged_lookup(diseases, 'text', 'extracted_diseases', 'disease texts')(pl.Series(['asthma']))
print(sorted({module.split('.')[0] for module in sys.modules} & {'openai', 'torch', 'transformers', 'huggingface_hub'}))
"""


TRIAL_MODELS_STEP = """
import sys

import pts.transformers.clinical_mira.aact_trial_models
import pts.postgres
import pts.transformers.clinical_mira.hf_models
import pts.transformers.clinical_mira.ner
import pts.transformers.clinical_mira.providers.aact

print(sorted({module.split('.')[0] for module in sys.modules} & {'openai'}))
"""


def test_trial_models_step_imports_no_openai() -> None:
    result = subprocess.run(
        [sys.executable, '-c', TRIAL_MODELS_STEP],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[-1] == '[]'


def test_release_imports_no_torch_or_openai() -> None:
    result = subprocess.run(
        [sys.executable, '-c', RELEASE_MAPPING],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[-1] == '[]'


def test_fda_reports_keep_their_drug_without_reaching_ner() -> None:
    application = {
        'application_number': 'NDA017001',
        'products': [{'marketing_status': 'Prescription', 'active_ingredients': [{'name': 'ASPIRIN'}]}],
        'submissions': [{'submission_type': 'ORIG', 'submission_status': 'AP', 'submission_status_date': '20240314'}],
    }
    fda = extract_fda_clinical_report([application], DRUG).df.with_columns(
        drugs=pl.col('drugs').list.eval(pl.element().struct.with_fields(drugFromSource=pl.lit('no dictionary name')))
    )

    mapped = map_report_entities(fda, DISEASE, DRUG, chembl_curation=None, drug_entities=_staged_drug_entities())
    flagged = flag_null_diseases(mapped)

    (row,) = flagged.df.iter_rows(named=True)
    assert row['drugs'] == [{'drugFromSource': 'no dictionary name', 'drugId': 'CHEMBL25'}]
    assert 'NO_DISEASE' in row['qualityControls']
