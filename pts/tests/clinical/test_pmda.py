import polars as pl
import pytest

from pts.transformers.clinical.providers.pmda import extract_approval_year, find_column_structure
from pts.transformers.clinical.providers.pmda import extract_clinical_report as extract_pmda_clinical_report


@pytest.mark.parametrize(
    ('value', 'expected'),
    [
        ('Sep. 24, 2024', 2024),
        ('May 7, 2020', 2020),
        ('20-Apr-06', 2006),
        ('15-Jun-06', 2006),
        ('(1) Aug. 21,\n2018\n(2) Aug. 22,\n2018', 2018),
        ('', None),
        (None, None),
        ('2266--JJuull--0066', None),
    ],
)
def test_extract_approval_year(value, expected) -> None:
    assert extract_approval_year(value) == expected


def test_extract_approval_year_in_select() -> None:
    df = pl.DataFrame({'approval_date': ['Sep. 24, 2024', '', '20-Apr-06', None]})
    out = df.with_columns(year=pl.col('approval_date').map_elements(extract_approval_year, return_dtype=pl.Int32))
    assert out['year'].to_list() == [2024, None, 2006, None]


def test_find_column_structure_detects_reversed_date_header() -> None:
    """Pages 185-196 use 'Date of approval' instead of 'Approval Date'."""
    header = [
        'Category',
        'Date of\napproval',
        'Brand name (name of company)',
        'Approval/ Supplemental Change',
        'Names of ingredients (Underlined: New active ingredients)',
        'Note',
    ]
    structure = find_column_structure(header)
    assert structure.approval_date_idx == 1
    assert structure.brand_name_idx == 2
    assert structure.ingredient_idx == 4


def test_extract_approval_year_two_digit_dd_mon_yy() -> None:
    """Two-digit year format like '9-Jul-04' maps to the 2000s."""
    assert extract_approval_year('9-Jul-04') == 2004


def _approvals(rows: list[tuple[str, str | None, str | None]]) -> pl.DataFrame:
    """Approvals as parsed from the PDF: (approval_date, active_ingredient, notes)."""
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


def test_pmda_provider_uses_injected_ner() -> None:
    approvals = _approvals([
        # Two ingredients and two diseases: one report per pair
        ('Sep. 24, 2024', '(1) Alphamab\n/ (2) Betanib (genetical recombination)', ' Asthma and COPD '),
        # Same note text: NER runs once for both
        ('May 7, 2020', 'Gammastat', 'Asthma and COPD'),
        # NER finds nothing: no report
        ('20-Apr-06', 'Deltastat', 'Nothing here'),
        # Null note reaches the NER as an empty text: no report
        ('20-Apr-06', 'Epsilonib', None),
        # N/A ingredient: no report
        ('20-Apr-06', 'N/A', 'Asthma and COPD'),
    ])
    calls: list[list[str]] = []

    def extract_diseases(texts: pl.Series) -> pl.Series:
        calls.append(texts.to_list())
        extracted = {'Asthma and COPD': ['asthma', 'copd'], 'Nothing here': [], '': []}
        return pl.Series([extracted[t] for t in texts], dtype=pl.List(pl.String))

    reports = extract_pmda_clinical_report(approvals, extract_diseases).df

    assert calls == [['', 'Asthma and COPD', 'Nothing here']]
    rows = sorted(
        (
            r['drugs'][0]['drugFromSource'],
            r['diseases'][0]['diseaseFromSource'],
            r['year'],
            len(r['drugs']),
            len(r['diseases']),
        )
        for r in reports.iter_rows(named=True)
    )
    assert rows == [
        ('alphamab', 'asthma', 2024, 1, 1),
        ('alphamab', 'copd', 2024, 1, 1),
        ('betanib', 'asthma', 2024, 1, 1),
        ('betanib', 'copd', 2024, 1, 1),
        ('gammastat', 'asthma', 2020, 1, 1),
        ('gammastat', 'copd', 2020, 1, 1),
    ]
    assert set(reports['provider']) == {'PMDA'}
    assert set(reports['source']) == {'PMDA'}
    assert set(reports['clinicalStage']) == {'APPROVAL'}
    assert reports['id'].n_unique() == reports.height
