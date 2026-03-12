import polars as pl
from clinical_mining.dataset import ClinicalReport

from pts.pyspark.clinical_report import validate_disease


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
        'type': 'CURATED_RESOURCE',
        'source': 'DailyMed',
        'year': None,
        'countries': ['United States'],
        'hasExpertReview': True,
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
