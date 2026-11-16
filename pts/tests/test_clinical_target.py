"""Tests for the clinical_target step."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from pts.transformers.clinical_target import clinical_target

REPORT_SCHEMA = {
    'id': pl.String,
    'clinicalStage': pl.String,
    'drugs': pl.List(pl.Struct({'drugFromSource': pl.String, 'drugId': pl.String})),
    'diseases': pl.List(pl.Struct({'diseaseFromSource': pl.String, 'diseaseId': pl.String})),
    'qualityControls': pl.List(pl.String),
}
MOA_SCHEMA = {'chemblIds': pl.List(pl.String), 'targets': pl.List(pl.String), 'curation': pl.String}


def _report(report_id: str, drug_id: str) -> dict:
    return {
        'id': report_id,
        'clinicalStage': 'PHASE_2',
        'drugs': [{'drugFromSource': drug_id.lower(), 'drugId': drug_id}],
        'diseases': [{'diseaseFromSource': 'Asthma', 'diseaseId': 'EFO_1'}],
        'qualityControls': [],
    }


def test_a_drug_with_only_an_electronic_mechanism_gets_a_clinical_target_row(tmp_path: Path) -> None:
    """CHEMBL9 has no ChEMBL mechanism; its electronic row yields a clinical target like a manual one."""
    (tmp_path / 'report').mkdir()
    (tmp_path / 'moa').mkdir()
    reports = [_report('R1', 'CHEMBL1'), _report('R2', 'CHEMBL9')]
    pl.DataFrame(reports, schema=REPORT_SCHEMA).write_parquet(tmp_path / 'report' / 'part.parquet')
    moa = [
        {'chemblIds': ['CHEMBL1'], 'targets': ['ENSG1'], 'curation': 'manual'},
        {'chemblIds': ['CHEMBL9'], 'targets': ['ENSG9'], 'curation': 'electronic'},
    ]
    pl.DataFrame(moa, schema=MOA_SCHEMA).write_parquet(tmp_path / 'moa' / 'part.parquet')

    clinical_target(
        {'clinical_report': tmp_path / 'report', 'drug_mechanism_of_action': tmp_path / 'moa'},
        {'output': tmp_path / 'out', 'excluded': tmp_path / 'excluded'},
        {},
        None,  # type: ignore[arg-type]
    )

    got = pl.read_parquet(tmp_path / 'out' / '*.parquet').sort('drugId')
    assert got.select('drugId', 'targetId', 'clinicalReportIds', 'maxClinicalStage').rows() == [
        ('CHEMBL1', 'ENSG1', ['R1'], 'PHASE_2'),
        ('CHEMBL9', 'ENSG9', ['R2'], 'PHASE_2'),
    ]
