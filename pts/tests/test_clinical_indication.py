"""Tests for the clinical_indication step."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from pts.transformers.clinical_indication import clinical_indication

REPORT_SCHEMA = {
    'id': pl.String,
    'clinicalStage': pl.String,
    'origin': pl.String,
    'type': pl.String,
    'source': pl.String,
    'provider': pl.String,
    'drugs': pl.List(pl.Struct({'drugFromSource': pl.String, 'drugId': pl.String})),
    'diseases': pl.List(pl.Struct({'diseaseFromSource': pl.String, 'diseaseId': pl.String})),
    'qualityControls': pl.List(pl.String),
}
OUTPUT_COLUMNS = ['id', 'maxClinicalStage', 'clinicalReportIds', 'drugId', 'diseaseId']


def _report(
    report_id: str,
    drugs: list[str | None] | None = None,
    diseases: list[str | None] | None = None,
    stage: str = 'PHASE_2',
    quality_controls: list[str] | None = None,
) -> dict:
    drugs = ['CHEMBL1'] if drugs is None else drugs
    diseases = ['EFO_1'] if diseases is None else diseases
    return {
        'id': report_id,
        'clinicalStage': stage,
        'origin': 'CLINICAL_TRIAL',
        'type': 'INDICATION',
        'source': 'ClinicalTrials.gov',
        'provider': 'AACT',
        'drugs': [{'drugFromSource': f'drug {i}', 'drugId': drug_id} for i, drug_id in enumerate(drugs)],
        'diseases': [
            {'diseaseFromSource': f'disease {i}', 'diseaseId': disease_id} for i, disease_id in enumerate(diseases)
        ],
        'qualityControls': quality_controls,
    }


def _run(tmp_path: Path, reports: list[dict], settings: dict | None = None) -> tuple[pl.DataFrame, pl.DataFrame]:
    (tmp_path / 'report').mkdir()
    pl.DataFrame(reports, schema=REPORT_SCHEMA).write_parquet(tmp_path / 'report' / 'part.parquet')

    clinical_indication(
        tmp_path / 'report',
        {'output': tmp_path / 'out', 'excluded': tmp_path / 'excluded'},
        settings or {},
        None,  # type: ignore[arg-type]
    )

    return (
        pl.read_parquet(tmp_path / 'out' / '*.parquet'),
        pl.read_parquet(tmp_path / 'excluded' / '*.parquet'),
    )


def _pairs(indications: pl.DataFrame) -> list[tuple]:
    return (
        indications
        .select('drugId', 'diseaseId', pl.col('clinicalReportIds').list.sort(), 'maxClinicalStage')
        .sort('drugId', 'diseaseId')
        .rows()
    )


def test_only_fully_mapped_indications_are_written(tmp_path: Path) -> None:
    reports = [
        _report('R1'),
        _report('R2', drugs=[None]),
        _report('R3', diseases=[None]),
        _report('R4', drugs=[None], diseases=[None]),
    ]

    indications, _ = _run(tmp_path, reports)

    assert indications.columns == OUTPUT_COLUMNS
    assert _pairs(indications) == [('CHEMBL1', 'EFO_1', ['R1'], 'PHASE_2')]


def test_reports_on_the_same_drug_and_disease_merge_at_the_most_advanced_stage(tmp_path: Path) -> None:
    reports = [
        _report('R1', stage='PHASE_1'),
        _report('R2', stage='PHASE_3'),
        _report('R3', stage='UNKNOWN'),
    ]

    indications, _ = _run(tmp_path, reports)

    assert _pairs(indications) == [('CHEMBL1', 'EFO_1', ['R1', 'R2', 'R3'], 'PHASE_3')]


@pytest.mark.parametrize('stage', ['APPROVAL', 'PHASE_4', 'WITHDRAWAL'])
def test_post_approval_stages_are_reported_as_approval(tmp_path: Path, stage: str) -> None:
    reports = [_report('R1', stage='PHASE_3'), _report('R2', stage=stage)]

    indications, _ = _run(tmp_path, reports)

    assert _pairs(indications) == [('CHEMBL1', 'EFO_1', ['R1', 'R2'], 'APPROVAL')]


def test_a_report_with_several_drugs_and_diseases_supports_every_pair(tmp_path: Path) -> None:
    reports = [_report('R1', drugs=['CHEMBL1', 'CHEMBL2'], diseases=['EFO_1', 'EFO_2'])]

    indications, _ = _run(tmp_path, reports)

    assert _pairs(indications) == [
        ('CHEMBL1', 'EFO_1', ['R1'], 'PHASE_2'),
        ('CHEMBL1', 'EFO_2', ['R1'], 'PHASE_2'),
        ('CHEMBL2', 'EFO_1', ['R1'], 'PHASE_2'),
        ('CHEMBL2', 'EFO_2', ['R1'], 'PHASE_2'),
    ]


def test_the_id_depends_only_on_the_drug_and_disease(tmp_path: Path) -> None:
    reports = [
        _report('R1', drugs=['CHEMBL1'], diseases=['EFO_1']),
        _report('R2', drugs=['CHEMBL1'], diseases=['EFO_2'], stage='PHASE_3'),
        _report('R3', drugs=['CHEMBL1'], diseases=['EFO_1'], stage='PHASE_1'),
    ]

    indications, _ = _run(tmp_path, reports)

    assert indications.height == 2
    assert indications['id'].n_unique() == 2
    assert indications['id'].str.len_chars().to_list() == [64, 64]


def test_reports_failing_an_invalid_quality_control_are_excluded(tmp_path: Path) -> None:
    reports = [
        _report('R1', diseases=['EFO_1'], quality_controls=['PHASE_IV_NOT_APPROVED']),
        _report('R2', diseases=['EFO_2'], quality_controls=['SOME_OTHER_FLAG']),
        _report('R3', diseases=['EFO_3'], quality_controls=None),
        _report('R4', diseases=['EFO_4'], quality_controls=['SOME_OTHER_FLAG', 'INDIRECT_PRIMARY_PURPOSE']),
    ]
    settings = {'invalid_clinical_report_qc': ['PHASE_IV_NOT_APPROVED', 'INDIRECT_PRIMARY_PURPOSE']}

    indications, excluded = _run(tmp_path, reports, settings)

    assert sorted(excluded['id']) == ['R1', 'R4']
    assert excluded.columns == list(REPORT_SCHEMA)
    assert _pairs(indications) == [
        ('CHEMBL1', 'EFO_2', ['R2'], 'PHASE_2'),
        ('CHEMBL1', 'EFO_3', ['R3'], 'PHASE_2'),
    ]


def test_an_excluded_report_does_not_support_an_indication_it_shares(tmp_path: Path) -> None:
    reports = [
        _report('R1', stage='APPROVAL', quality_controls=['PHASE_IV_NOT_APPROVED']),
        _report('R2', stage='PHASE_2'),
    ]
    settings = {'invalid_clinical_report_qc': ['PHASE_IV_NOT_APPROVED']}

    indications, _ = _run(tmp_path, reports, settings)

    assert _pairs(indications) == [('CHEMBL1', 'EFO_1', ['R2'], 'PHASE_2')]


def test_without_invalid_quality_controls_nothing_is_excluded(tmp_path: Path) -> None:
    reports = [_report('R1', quality_controls=['PHASE_IV_NOT_APPROVED'])]

    indications, excluded = _run(tmp_path, reports)

    assert excluded.is_empty()
    assert excluded.columns == list(REPORT_SCHEMA)
    assert _pairs(indications) == [('CHEMBL1', 'EFO_1', ['R1'], 'PHASE_2')]
