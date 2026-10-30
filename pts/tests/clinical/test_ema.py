import polars as pl
import pytest

from pts.transformers.clinical.providers.ema import extract_clinical_report as extract_ema_clinical_report
from pts.transformers.clinical.providers.ema import extract_marketing_year

EMA_COLUMNS = [
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
]


def _ema_sheet(rows: list[dict]) -> pl.DataFrame:
    """Build the raw EMA sheet: generic column names, real names in the first row."""
    defaults = {
        'Category': 'Human',
        'Medicine status': 'Authorised',
        'Marketing authorisation date': '19/06/2015',
        'Medicine URL': 'https://www.ema.europa.eu/en/medicines/human/EPAR/x',
    }
    records = [dict(zip(EMA_COLUMNS, EMA_COLUMNS, strict=True))] + [
        {c: r.get(c, defaults.get(c)) for c in EMA_COLUMNS} for r in rows
    ]
    return pl.DataFrame(
        [[rec[c] for c in EMA_COLUMNS] for rec in records],
        schema=[f'column_{i}' for i in range(len(EMA_COLUMNS))],
        orient='row',
    )


def _stub_ner(extracted: dict[str, list[str]], calls: list[list[str]] | None = None):
    """Stand in for the disease NER: map each text to fixed diseases."""

    def extract_diseases(texts: pl.Series) -> pl.Series:
        if calls is not None:
            calls.append(texts.to_list())
        return pl.Series([extracted[text] for text in texts], dtype=pl.List(pl.String))

    return extract_diseases


def test_extract_clinical_report_keeps_medicine_when_ner_finds_nothing() -> None:
    """A medicine whose indication yields no NER disease is kept through its MeSH term.

    Polars 2.0 explodes an empty list into zero rows, which silently dropped
    these medicines; 1.x kept them as a null row that the MeSH coalesce fills.
    """
    sheet = _ema_sheet([
        {
            'Name of medicine': 'Alpha',
            'EMA product number': 'EMEA/H/C/000001',
            'International non-proprietary name (INN) / common name': 'alphamab',
            'Active substance': 'alphamab',
            'Therapeutic indication': 'treatment of breast cancer',
        },
        {
            # NER finds nothing, but the MeSH term identifies the disease
            'Name of medicine': 'Beta',
            'EMA product number': 'EMEA/H/C/000002',
            'International non-proprietary name (INN) / common name': 'betanib',
            'Active substance': 'betanib',
            'Therapeutic area (MeSH)': 'Diabetes Mellitus, Type 2',
            'Therapeutic indication': 'see product information',
        },
        {
            # Neither NER nor MeSH give a disease: no report
            'Name of medicine': 'Gamma',
            'EMA product number': 'EMEA/H/C/000003',
            'International non-proprietary name (INN) / common name': 'gammastat',
            'Active substance': 'gammastat',
            'Therapeutic indication': 'nothing to extract',
        },
    ])
    ner = _stub_ner({
        'treatment of breast cancer': ['breast cancer'],
        'see product information': [],
        'nothing to extract': [],
    })

    reports = extract_ema_clinical_report(sheet, ner).df

    assert sorted(reports['id'].to_list()) == ['emea/h/c/000001', 'emea/h/c/000002']
    diseases = {r['id']: [d['diseaseFromSource'] for d in r['diseases']] for r in reports.iter_rows(named=True)}
    assert diseases == {
        'emea/h/c/000001': ['breast cancer'],
        'emea/h/c/000002': ['diabetes mellitus, type 2'],
    }


def test_ema_provider_uses_injected_ner() -> None:
    sheet = _ema_sheet([
        {
            # MeSH wins over the NER result
            'Name of medicine': 'Alpha',
            'EMA product number': 'EMEA/H/C/000001',
            'Active substance': 'alphamab',
            'Therapeutic area (MeSH)': 'Breast Neoplasms;Neoplasm Metastasis',
            'Therapeutic indication': ' treatment of breast cancer ',
        },
        {
            # Same text as Beta: NER runs once for both; two diseases, two drugs
            'Name of medicine': 'Beta',
            'EMA product number': 'EMEA/H/C/000002',
            'International non-proprietary name (INN) / common name': 'betanib;betamab',
            'Therapeutic indication': 'asthma and copd',
        },
        {
            'Name of medicine': 'Delta',
            'EMA product number': 'EMEA/H/C/000004',
            'Active substance': 'deltastat',
            'Therapeutic indication': 'asthma and copd',
        },
        {
            # Null indication reaches the NER as an empty text, and finds nothing
            'Name of medicine': 'Gamma',
            'EMA product number': 'EMEA/H/C/000003',
            'Active substance': 'gammastat',
            'Therapeutic indication': None,
        },
        {
            # Not a human medicine
            'Category': 'Veterinary',
            'Name of medicine': 'Epsilon',
            'EMA product number': 'EMEA/V/C/000005',
            'Active substance': 'epsilonib',
            'Therapeutic indication': 'mastitis',
        },
    ])
    calls: list[list[str]] = []
    ner = _stub_ner(
        {
            'treatment of breast cancer': ['breast cancer'],
            'asthma and copd': ['asthma', 'copd'],
            '': [],
        },
        calls,
    )

    reports = extract_ema_clinical_report(sheet, ner).df

    assert calls == [['', 'asthma and copd', 'treatment of breast cancer']]
    rows = {
        r['id']: (
            sorted(d['diseaseFromSource'] for d in r['diseases']),
            sorted(d['drugFromSource'] for d in r['drugs']),
            r['provider'],
            r['source'],
            r['year'],
        )
        for r in reports.iter_rows(named=True)
    }
    assert rows == {
        'emea/h/c/000001': (
            ['breast neoplasms', 'neoplasm metastasis'],
            ['alphamab'],
            'EMA',
            'EMA Human Drugs',
            2015,
        ),
        'emea/h/c/000002': (['asthma', 'copd'], ['betamab', 'betanib'], 'EMA', 'EMA Human Drugs', 2015),
        'emea/h/c/000004': (['asthma', 'copd'], ['deltastat'], 'EMA', 'EMA Human Drugs', 2015),
    }
    assert all(d['diseaseId'] is None for r in reports['diseases'].to_list() for d in r)


def test_ema_provider_does_not_rename_the_callers_sheet() -> None:
    sheet = _ema_sheet([
        {
            'EMA product number': 'EMEA/H/C/000001',
            'Active substance': 'alphamab',
            'Therapeutic indication': 'x',
        },
    ])
    columns = sheet.columns

    extract_ema_clinical_report(sheet, _stub_ner({'x': ['x disease']}))

    assert sheet.columns == columns


def test_ema_provider_rejects_ner_of_the_wrong_length() -> None:
    sheet = _ema_sheet([
        {'EMA product number': 'EMEA/H/C/000001', 'Active substance': 'a', 'Therapeutic indication': 'x'},
    ])

    with pytest.raises(ValueError, match='extraction returned 0 rows for 1 texts'):
        extract_ema_clinical_report(sheet, lambda texts: pl.Series([], dtype=pl.List(pl.String)))


@pytest.mark.parametrize(
    ('value', 'expected'),
    [
        ('19/06/2015', 2015),
        ('08/03/2002', 2002),
        ('', None),
        (None, None),
        ('not a date', None),
    ],
)
def test_extract_marketing_year(value, expected) -> None:
    assert extract_marketing_year(value) == expected


def test_extract_marketing_year_in_select() -> None:
    df = pl.DataFrame({'Marketing authorisation date': ['19/06/2015', '', '01/01/2024']})
    out = df.with_columns(
        year=pl.col('Marketing authorisation date').map_elements(extract_marketing_year, return_dtype=pl.Int32)
    )
    assert out['year'].to_list() == [2015, None, 2024]
