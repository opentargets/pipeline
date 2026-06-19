import polars as pl
from clinical_mining.dataset import ClinicalReport

from pts.pyspark.clinical_report import (
    ClinicalReportFlags,
    flag_indirect_primary_purpose,
    validate_disease,
)


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


def _report(id_: str, primary_purpose: str | None = None) -> dict:
    return {
        'id': id_,
        'phaseFromSource': 'phase 3',
        'type': 'CURATED_RESOURCE',
        'source': 'ClinicalTrials',
        'trialPrimaryPurpose': primary_purpose,
        'drugs': [{'drugFromSource': 'BENAZEPRIL', 'drugId': 'CHEMBL1694'}],
        'diseases': [{'diseaseFromSource': 'hypertension', 'diseaseId': 'EFO:0000537'}],
    }


def _flagged(reports: ClinicalReport, report_id: str) -> bool:
    qc = reports.df.filter(pl.col('id') == report_id).select('qualityControls').to_series().to_list()[0]
    return qc is not None and ClinicalReportFlags.INDIRECT_PRIMARY_PURPOSE.value in qc


def test_flag_indirect_primary_purpose_device_feasibility() -> None:
    reports = _build_reports([
        _report('r1', primary_purpose='TREATMENT'),
        _report('r2', primary_purpose='DEVICE_FEASIBILITY'),
        _report('r3', primary_purpose='DIAGNOSTIC'),
        _report('r4', primary_purpose='OTHER'),
    ])
    result = flag_indirect_primary_purpose(reports)
    assert not _flagged(result, 'r1')
    assert _flagged(result, 'r2')
    assert _flagged(result, 'r3')
    assert not _flagged(result, 'r4')


def test_flag_indirect_primary_purpose_no_primary_purpose() -> None:
    """Reports without a trialPrimaryPurpose should not be flagged."""
    reports = _build_reports([
        _report('r1', primary_purpose=None),
    ])
    result = flag_indirect_primary_purpose(reports)
    assert not _flagged(result, 'r1')


def test_flag_indirect_primary_purpose_with_llm_no_match_is_flagged() -> None:
    """When llm_batch_results is provided but report has no match, drug_intent is null → flagged."""
    reports = _build_reports([
        _report('r1', primary_purpose='TREATMENT'),
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
        _report('r1', primary_purpose='TREATMENT'),
    ])
    llm_batch_results = pl.DataFrame({'id': ['r1'], 'drug_intent': ['therapeutic']})
    result = flag_indirect_primary_purpose(reports, llm_drug_intent=llm_batch_results)
    assert not _flagged(result, 'r1')


def test_flag_indirect_primary_purpose_with_llm_non_therapeutic_flagged() -> None:
    """Non-therapeutic drug_intent values (prevention, supportive_care, etc.) should be flagged."""
    reports = _build_reports([
        _report('r1', primary_purpose='TREATMENT'),
        _report('r2', primary_purpose='TREATMENT'),
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
        _report('r1', primary_purpose='TREATMENT'),
    ])
    llm_batch_results = pl.DataFrame({'id': ['r1'], 'drug_intent': ['therapeutic']})
    result = flag_indirect_primary_purpose(reports, llm_drug_intent=llm_batch_results)
    assert 'drug_intent' not in result.df.columns
