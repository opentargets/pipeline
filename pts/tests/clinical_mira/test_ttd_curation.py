import polars as pl

from pts.transformers.clinical_mira.providers.ttd import extract_clinical_report, extract_indication, ttd_report_id

TTD_LINES = [
    'TTDDRUID\tD0A001',
    'DRUGNAME\tAlphamab',
    'INDICATI\tplaceholder row the parser drops\tICD-11: XX00\tApproved',
    'INDICATI\tAge-related Macular Degeneration\tICD-11: 9B75\tApproved',
    'TTDDRUID\tD0B002',
    'DRUGNAME\tBetanib',
    'INDICATI\tChronic obstructive pulmonary disease\tICD-11: CA22\tPhase 2',
    'INDICATI\tAsthma\tICD-11: CA23\tPhase 3',
]


def test_ttd_report_id_matches_extracted_report_ids() -> None:
    indications = extract_indication(TTD_LINES)

    reports = extract_clinical_report(indications)
    expected = indications.select(ttd_report_id(pl.col('ttd_id'), pl.col('diseaseFromSource')).alias('id'))

    assert set(reports.df['id']) == set(expected['id'])
    assert len(set(expected['id'])) == 3
