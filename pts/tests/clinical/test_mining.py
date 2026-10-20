from pathlib import Path

import openpyxl
import polars as pl
import pytest

from pts.transformers.clinical import (
    ClinicalReport,
    extract_ema_clinical_report,
    extract_pmda_clinical_report,
    extract_ttd_clinical_report,
    extract_ttd_indication,
)
from pts.transformers.clinical import mining as mining_module
from pts.transformers.clinical.mining import (
    RecordingNer,
    build_mining_reports,
    read_ema_sheet,
    report_drug_labels,
    staged_lookup,
)
from tests.test_clinical_report import EMA_BANNER, EMA_HEADER, EMA_NER, _ema_workbook

TTD_LINES = [
    'TTDDRUID\tD0A001',
    'DRUGNAME\tAlphamab',
    'INDICATI\tplaceholder row the parser drops\tICD-11: XX00\tApproved',
    'INDICATI\tAsthma\tICD-11: CA23\tApproved',
    'TTDDRUID\tD0B002',
    'DRUGNAME\tBetanib',
    'INDICATI\tChronic obstructive pulmonary disease\tICD-11: CA22\tPhase 2',
    'INDICATI\tAsthma\tICD-11: CA23\tPhase 3',
    '',
]

PMDA_NER = {
    'for the treatment of asthma': ['asthma'],
    'for the treatment of gout': ['gout'],
}


def _approvals(rows: list[tuple[str, str | None, str | None]]) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                'approval_date': date,
                'brand_name': 'Brand',
                'active_ingredient': ingredient,
                'notes': notes,
                'approval_type': 'Approval',
                'page_number': page,
            }
            for page, (date, ingredient, notes) in enumerate(rows, 1)
        ],
        schema={
            'approval_date': pl.String,
            'brand_name': pl.String,
            'active_ingredient': pl.String,
            'notes': pl.String,
            'approval_type': pl.String,
            'page_number': pl.Int64,
        },
    )


APPROVALS = _approvals([
    ('Sep. 24, 2024', 'Alphamab', ' for the treatment of asthma '),
    ('May 7, 2020', 'Gammastat', 'for the treatment of gout'),
    ('20-Apr-06', 'Deltanib', None),
    ('20-Apr-06', 'Epsilonib', '   '),
])


def _ner(texts: pl.Series) -> pl.Series:
    known = {**EMA_NER, **PMDA_NER}
    return pl.Series([known.get(text, []) for text in texts], dtype=pl.List(pl.String))


@pytest.fixture
def inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """TTD and EMA inputs on disk; the PMDA PDF is a stub file, its parser replaced by `APPROVALS`.

    A real approvals PDF is impractical as a fixture, so the PDF boundary (`pdfplumber.open` and
    `parse_pmda_approvals`) is patched in `mining`; the bytes are still read through the storage handle.
    """
    ttd = tmp_path / 'ttd.txt'
    ttd.write_text('\n'.join(TTD_LINES))
    pmda = tmp_path / 'pmda.pdf'
    pmda.write_bytes(b'%PDF-stub')
    monkeypatch.setattr(mining_module.pdfplumber, 'open', lambda _: object())
    monkeypatch.setattr(mining_module, 'parse_pmda_approvals', lambda _: APPROVALS)
    return {'ttd': str(ttd), 'ema': _ema_workbook(tmp_path / 'ema.xlsx'), 'pmda': str(pmda)}


def _comparable(report: ClinicalReport) -> list[dict]:
    return report.df.sort('id').with_columns(pl.col('drugs', 'diseases').list.sort()).to_dicts()


def test_build_mining_reports_matches_the_release_path(inputs: dict[str, str]) -> None:
    expected = {
        'ttd': extract_ttd_clinical_report(indications=extract_ttd_indication(TTD_LINES)),
        'ema': extract_ema_clinical_report(read_ema_sheet(inputs['ema']), extract_diseases=_ner),
        'pmda': extract_pmda_clinical_report(df=APPROVALS, extract_diseases=_ner),
    }

    reports = build_mining_reports(inputs['ttd'], inputs['ema'], inputs['pmda'], _ner)

    assert sorted(reports) == ['ema', 'pmda', 'ttd']
    for name, report in reports.items():
        assert report.df.height > 0
        assert _comparable(report) == _comparable(expected[name])


def test_recorded_disease_texts_are_what_the_providers_pass(tmp_path: Path, inputs: dict[str, str]) -> None:
    """Blank and whitespace-only indication texts reach the NER as '' and must be staged as such."""
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = 'Medicine'
    column = EMA_HEADER.index('Therapeutic indication')
    rows = [EMA_BANNER, EMA_HEADER]
    for number, text in enumerate(['  treatment of asthma  ', None, '   ', 'treatment of copd'], 1):
        row = ['Human', f'Name{number}', f'EMEA/H/C/00000{number}', 'Authorised'] + [None] * (len(EMA_HEADER) - 4)
        row[column] = text
        rows.append(row)
    for i, row in enumerate(rows, 1):
        for j, value in enumerate(row, 1):
            sheet.cell(row=i, column=j, value=value)
    workbook.save(tmp_path / 'blanks.xlsx')

    recorder = RecordingNer()
    build_mining_reports(inputs['ttd'], str(tmp_path / 'blanks.xlsx'), inputs['pmda'], recorder)

    assert recorder.texts == {
        'treatment of asthma',
        'treatment of copd',
        '',
        'for the treatment of asthma',
        'for the treatment of gout',
    }


def test_recording_ner_finds_nothing_and_keeps_the_texts() -> None:
    recorder = RecordingNer()

    out = recorder(pl.Series('t', ['a', '', ' ']))

    assert out.dtype == pl.List(pl.String)
    assert out.to_list() == [[], [], []]
    assert recorder.texts == {'a', '', ' '}


def test_staged_lookup_maps_texts_to_their_staged_values() -> None:
    staged = pl.DataFrame({'text': ['a', 'b', ''], 'diseases': [['x'], [], ['y', 'z']]})

    out = staged_lookup(staged, 'text', 'diseases', 'disease texts')(pl.Series('t', ['', 'a', 'b']))

    assert out.dtype == pl.List(pl.String)
    assert out.to_list() == [['y', 'z'], ['x'], []]


def test_staged_lookup_fails_on_a_missing_text() -> None:
    staged = pl.DataFrame({'text': ['a'], 'diseases': [['x']]})
    lookup = staged_lookup(staged, 'text', 'diseases', 'disease texts')

    with pytest.raises(ValueError, match=r"2 disease texts not staged, e\.g\. \['', 'zzz'\]"):
        lookup(pl.Series('t', ['a', 'zzz', '']))


def test_staged_lookup_fails_on_a_null_staged_value() -> None:
    staged = pl.DataFrame({'text': ['a'], 'diseases': pl.Series([None], dtype=pl.List(pl.String))})

    with pytest.raises(ValueError, match='1 disease texts not staged'):
        staged_lookup(staged, 'text', 'diseases', 'disease texts')(pl.Series('t', ['a']))


def test_report_drug_labels_skips_labels_with_an_id() -> None:
    drug = pl.Struct({'drugFromSource': pl.String, 'drugId': pl.String})
    first = pl.DataFrame(
        {
            'drugs': [
                [{'drugFromSource': 'zeta', 'drugId': None}, {'drugFromSource': 'known', 'drugId': 'CHEMBL1'}],
                [{'drugFromSource': 'alpha', 'drugId': None}],
            ]
        },
        schema={'drugs': pl.List(drug)},
    )
    second = pl.DataFrame(
        {'drugs': [[{'drugFromSource': 'alpha', 'drugId': None}, {'drugFromSource': None, 'drugId': None}]]},
        schema={'drugs': pl.List(drug)},
    )

    assert report_drug_labels([first, second]).to_list() == ['alpha', 'zeta']
    assert report_drug_labels([]).to_list() == []
