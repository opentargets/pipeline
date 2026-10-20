import datetime
import socket
from pathlib import Path

import huggingface_hub.constants as hf_constants
import openpyxl
import polars as pl
import pytest
from huggingface_hub import hf_hub_download

from pts.transformers.clinical import (
    ClinicalProvider,
    ClinicalReport,
    ClinicalReportOrigin,
    ClinicalReportType,
    ClinicalSource,
    extract_ema_clinical_report,
    replace_with_llm_indications,
)
from pts.transformers.clinical import mapping as mapping_module
from pts.transformers.clinical_report import (
    ClinicalReportFlags,
    flag_indirect_primary_purpose,
    map_report_entities,
    offline_hf_models,
    order_output_columns,
    read_ema_sheet,
    validate_disease,
)
from tests.clinical.mapping_fixtures import DISEASE, DRUG


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
        'drug_intent': ['therapeutic', 'prevention'],
    })
    result = flag_indirect_primary_purpose(reports, llm_drug_intent=llm_batch_results)
    assert not _flagged(result, 'r1')
    assert _flagged(result, 'r2')


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


def _extraction(nct_id: str, disease: str, drug: str) -> pl.DataFrame:
    """A minimal frame in `ClinicalReportExtractionSchema` shape, as `parse_batch_results` returns."""
    return pl.DataFrame({
        'id': [nct_id],
        'drug_intent': ['therapeutic'],
        'drug_intent_confidence': [0.99],
        'primary_indications': [[{'name': disease, 'evidence_quote': 'q'}]],
        'investigated_drugs': [[{'drug': drug, 'evidence_quote': 'q'}]],
    })


def test_llm_extractions_are_passed_in_their_raw_schema_shape() -> None:
    """`clinical_report` must hand the extraction to the AACT provider unprojected."""
    studies = pl.DataFrame({
        'nct_id': ['NCT0001'],
        'diseaseFromSource': ['from-aact'],
        'drugFromSource': ['from-aact'],
    })

    result = replace_with_llm_indications(studies, _extraction('NCT0001', 'asthma', 'aspirin'))

    assert result['diseaseFromSource'].to_list() == ['asthma']
    assert result['drugFromSource'].to_list() == ['aspirin']

    pre_projected = pl.DataFrame({'id': ['NCT0001'], 'drugs': [['aspirin']], 'diseases': [['asthma']]})
    with pytest.raises(pl.exceptions.ColumnNotFoundError, match='primary_indications'):
        replace_with_llm_indications(studies, pre_projected)


def test_offline_hf_models_serves_staged_files_without_the_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    revision = 'a' * 40
    staged = tmp_path / 'models' / 'some-org' / 'some-model' / revision
    staged.mkdir(parents=True)
    (staged / 'config.json').write_text('{"staged": true}')

    def no_network(*args, **kwargs):
        raise AssertionError('network access attempted')

    monkeypatch.setattr(socket, 'create_connection', no_network)
    previous = hf_constants.HF_HUB_OFFLINE, hf_constants.HF_HUB_CACHE

    with offline_hf_models(str(tmp_path / 'models')) as cache:
        assert hf_constants.HF_HUB_OFFLINE
        path = Path(hf_hub_download('some-org/some-model', 'config.json'))
        assert path.read_text() == '{"staged": true}'
        assert path.resolve().is_relative_to(cache.resolve())
        assert (cache / 'models--some-org--some-model' / 'refs' / 'main').read_text() == revision

    assert previous == (hf_constants.HF_HUB_OFFLINE, hf_constants.HF_HUB_CACHE)


def test_offline_hf_models_fails_when_nothing_is_staged(tmp_path: Path) -> None:
    (tmp_path / 'models').mkdir()

    with pytest.raises(ValueError, match='no models staged'), offline_hf_models(str(tmp_path / 'models')):
        pass


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


def test_map_report_entities_maps_each_drug_and_disease_of_a_report(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_ner(labels: pl.Series, batch_size: int) -> pl.Series:
        raise AssertionError(f'drug NER run on {labels.to_list()}')

    monkeypatch.setattr(mapping_module, 'extract_drug_entities', no_ner)
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

    mapped = map_report_entities(reports, DISEASE, DRUG, chembl_curation=None, ner_batch_size=8, ner_cache_path=None)

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
