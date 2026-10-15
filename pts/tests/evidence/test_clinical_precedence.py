"""Tests for clinical precedence evidence."""

from __future__ import annotations

import datetime
from pathlib import Path

import polars as pl
import pytest

from pts.transformers.evidence.clinical_precedence import (
    CLINICAL_PRECEDENCE,
    DIRECTION_ON_TARGET,
    DIRECTION_ON_TRAIT,
    clinical_precedence,
    generate,
    result_literature,
)
from tests.evidence.fixtures import write_disease, write_literature, write_target

INVALID_QC = ['PHASE_IV_NOT_APPROVED', 'INDIRECT_PRIMARY_PURPOSE']

REPORT_SCHEMA = {
    'id': pl.String,
    'clinicalStage': pl.String,
    'drugs': pl.List(pl.Struct({'drugFromSource': pl.String, 'drugId': pl.String})),
    'diseases': pl.List(pl.Struct({'diseaseFromSource': pl.String, 'diseaseId': pl.String})),
    'trialLiterature': pl.List(pl.Struct({'id': pl.String, 'type': pl.String})),
    'trialWhyStopped': pl.String,
    'qualityControls': pl.List(pl.String),
    'year': pl.Int32,
    'trialStopReasonCategories': pl.List(pl.String),
    'trialStartDate': pl.Date,
}
MOA_SCHEMA = {'actionType': pl.String, 'chemblIds': pl.List(pl.String), 'targets': pl.List(pl.String)}

RAW_SCHEMA = {
    'clinicalReportId': pl.String,
    'clinicalStage': pl.String,
    'trialWhyStopped': pl.String,
    'trialStopReasonCategories': pl.List(pl.String),
    'studyStartDate': pl.String,
    'evidenceDate': pl.String,
    'literature': pl.List(pl.String),
    'diseaseFromSource': pl.String,
    'diseaseFromSourceMappedId': pl.String,
    'drugFromSource': pl.String,
    'drugId': pl.String,
    'targetFromSourceId': pl.String,
    'datasourceId': pl.String,
    'datatypeId': pl.String,
    'actionType': pl.String,
}

ASTHMA = {'diseaseFromSource': 'Asthma', 'diseaseId': 'EFO_1'}
FLU = {'diseaseFromSource': 'Flu', 'diseaseId': 'EFO_2'}


def _report(report_id: str, diseases: list | None, drugs: list | None, **fields: object) -> dict:
    return {'id': report_id, 'clinicalStage': 'PHASE_3', 'diseases': diseases, 'drugs': drugs, **fields}


def _drug(name: str, drug_id: str | None) -> dict:
    return {'drugFromSource': name, 'drugId': drug_id}


REPORTS = [
    _report(
        'R1',
        [ASTHMA],
        [_drug('drugA', 'CHEMBL1'), _drug('drugA', 'CHEMBL1')],
        trialLiterature=[
            {'id': '111', 'type': 'RESULT'},
            {'id': '222', 'type': 'BACKGROUND'},
            {'id': '333', 'type': 'DERIVED'},
        ],
        year=2001,
        trialStartDate=datetime.date(2001, 2, 3),
    ),
    _report(
        'R2',
        [FLU, {'diseaseFromSource': 'Unmapped', 'diseaseId': None}, None],
        [_drug('drugB', 'CHEMBL2'), _drug('drugX', None)],
        clinicalStage='PHASE_2',
        trialWhyStopped='Low accrual',
        trialStopReasonCategories=['Insufficient_Enrollment'],
        trialLiterature=[{'id': None, 'type': 'RESULT'}],
        qualityControls=['OTHER_QC'],
    ),
    _report('R3', [], [_drug('drugA', 'CHEMBL1')]),
    _report('R4', None, [_drug('drugA', 'CHEMBL1')]),
    _report('R5', [ASTHMA], None),
    _report('R6', [ASTHMA], [_drug('drugA', 'CHEMBL1')], qualityControls=['PHASE_IV_NOT_APPROVED']),
    _report('R7', [ASTHMA], [_drug('drugA', 'CHEMBL1')], qualityControls=[None, 'INDIRECT_PRIMARY_PURPOSE']),
    _report(
        'R8',
        [ASTHMA],
        [_drug('drugC', 'CHEMBL3'), _drug('drugD', 'CHEMBL9'), _drug('drugE', 'CHEMBL4')],
        clinicalStage='APPROVAL',
        trialLiterature=[{'id': '444', 'type': 'BACKGROUND'}],
        qualityControls=[None],
        year=2010,
        trialStartDate=datetime.date(2009, 12, 31),
    ),
]

MECHANISMS = [
    {'actionType': 'INHIBITOR', 'chemblIds': ['CHEMBL1', 'CHEMBL2'], 'targets': ['ENSG1', 'ENSG2']},
    {'actionType': 'INHIBITOR', 'chemblIds': ['CHEMBL1'], 'targets': ['ENSG1']},
    {'actionType': 'AGONIST', 'chemblIds': ['CHEMBL1'], 'targets': ['ENSG3']},
    {'actionType': 'BLOCKER', 'chemblIds': ['CHEMBL3'], 'targets': None},
    {'actionType': 'BLOCKER', 'chemblIds': ['CHEMBL3'], 'targets': []},
    {'actionType': 'BLOCKER', 'chemblIds': None, 'targets': ['ENSG4']},
    {'actionType': 'BLOCKER', 'chemblIds': [], 'targets': ['ENSG4']},
    {'actionType': None, 'chemblIds': ['CHEMBL4'], 'targets': [None, 'ENSG5']},
]


def _row(report: str, stage: str, disease: dict, drug: dict, target: str, action: str | None, **fields: object) -> dict:
    return {
        'clinicalReportId': report,
        'clinicalStage': stage,
        'trialWhyStopped': None,
        'trialStopReasonCategories': None,
        'studyStartDate': None,
        'evidenceDate': None,
        'literature': None,
        'diseaseFromSource': disease['diseaseFromSource'],
        'diseaseFromSourceMappedId': disease['diseaseId'],
        'drugFromSource': drug['drugFromSource'],
        'drugId': drug['drugId'],
        'targetFromSourceId': target,
        'datasourceId': 'clinical_precedence',
        'datatypeId': 'clinical',
        'actionType': action,
        **fields,
    }


_R1 = {'studyStartDate': '2001-02-03', 'evidenceDate': '2001', 'literature': ['111', '333']}
_R2 = {'trialWhyStopped': 'Low accrual', 'trialStopReasonCategories': ['Insufficient_Enrollment']}
EXPECTED = [
    _row('R1', 'PHASE_3', ASTHMA, _drug('drugA', 'CHEMBL1'), 'ENSG1', 'INHIBITOR', **_R1),
    _row('R1', 'PHASE_3', ASTHMA, _drug('drugA', 'CHEMBL1'), 'ENSG2', 'INHIBITOR', **_R1),
    _row('R1', 'PHASE_3', ASTHMA, _drug('drugA', 'CHEMBL1'), 'ENSG3', 'AGONIST', **_R1),
    _row('R2', 'PHASE_2', FLU, _drug('drugB', 'CHEMBL2'), 'ENSG1', 'INHIBITOR', **_R2),
    _row('R2', 'PHASE_2', FLU, _drug('drugB', 'CHEMBL2'), 'ENSG2', 'INHIBITOR', **_R2),
    _row(
        'R8',
        'APPROVAL',
        ASTHMA,
        _drug('drugE', 'CHEMBL4'),
        'ENSG5',
        None,
        studyStartDate='2009-12-31',
        evidenceDate='2010',
    ),
]


def _sources(src: Path, reports: list[dict] = REPORTS) -> dict:
    (src / 'clinical_report').mkdir(parents=True)
    (src / 'moa').mkdir(parents=True)
    pl.DataFrame(reports, schema=REPORT_SCHEMA).write_parquet(src / 'clinical_report' / 'part.parquet')
    pl.DataFrame(MECHANISMS, schema=MOA_SCHEMA).write_parquet(src / 'moa' / 'part.parquet')
    return {'clinical_report': str(src / 'clinical_report'), 'drug_mechanism_of_action': str(src / 'moa')}


def _sorted(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda r: (r['clinicalReportId'], r['drugId'], r['targetFromSourceId']))


def _literature(rows: list[list[dict] | None]) -> list[list[str] | None]:
    df = pl.DataFrame({'trialLiterature': rows}, schema={'trialLiterature': REPORT_SCHEMA['trialLiterature']})
    return df.select(result_literature(pl.col('trialLiterature')).alias('out'))['out'].to_list()


def test_result_literature_keeps_outcome_references_and_drops_background() -> None:
    """RESULT and DERIVED report the trial's outcome; BACKGROUND is what its authors cited."""
    rows = [[{'id': '111', 'type': 'RESULT'}, {'id': '222', 'type': 'BACKGROUND'}, {'id': '333', 'type': 'DERIVED'}]]
    assert _literature(rows) == [['111', '333']]


def test_result_literature_is_null_when_only_background() -> None:
    """A trial citing background literature only yields null, not an empty list."""
    assert _literature([[{'id': '222', 'type': 'BACKGROUND'}]]) == [None]


def test_result_literature_is_null_when_trial_literature_is_null_or_empty() -> None:
    """Non-trial reports (ChEMBL, EMA, PMDA, TTD) have no trialLiterature at all."""
    assert _literature([None, []]) == [None, None]


def test_result_literature_drops_null_pmids_types_and_elements() -> None:
    """An outcome reference with no PMID must not put a null in the list."""
    rows = [
        [{'id': None, 'type': 'RESULT'}, {'id': '111', 'type': 'RESULT'}, {'id': '222', 'type': None}, None],
        [{'id': None, 'type': 'RESULT'}],
    ]
    assert _literature(rows) == [['111'], None]


def test_result_literature_matches_reference_type_case_insensitively() -> None:
    """Reference types match regardless of case, so a case change upstream is not silent."""
    rows = [[{'id': '111', 'type': 'result'}, {'id': '222', 'type': 'background'}, {'id': '333', 'type': 'Derived'}]]
    assert _literature(rows) == [['111', '333']]


def test_generate_joins_reports_to_mechanisms_and_explodes_targets(tmp_path: Path) -> None:
    raw, _ = generate(**_sources(tmp_path), invalid_clinical_report_qc=INVALID_QC)
    raw = raw.collect()

    assert raw.schema == pl.Schema(RAW_SCHEMA)
    assert _sorted(raw.to_dicts()) == EXPECTED


def test_generate_drops_null_and_empty_lists_as_spark_explode_does(tmp_path: Path) -> None:
    """R3-R5 have empty or null diseases or drugs; CHEMBL3 has null and empty targets."""
    raw, _ = generate(**_sources(tmp_path), invalid_clinical_report_qc=INVALID_QC)
    raw = raw.collect()

    assert set(raw['clinicalReportId']) == {'R1', 'R2', 'R8'}
    assert 'CHEMBL3' not in raw['drugId'].to_list()
    assert raw.filter(pl.col('targetFromSourceId').is_null()).height == 0


def test_generate_excludes_reports_failing_quality_control(tmp_path: Path) -> None:
    _, excluded = generate(**_sources(tmp_path), invalid_clinical_report_qc=INVALID_QC)
    excluded = excluded.collect()

    assert excluded.schema == pl.Schema(REPORT_SCHEMA)
    assert sorted(excluded['id']) == ['R6', 'R7']


def test_generate_excludes_nothing_without_invalid_quality_controls(tmp_path: Path) -> None:
    raw, excluded = generate(**_sources(tmp_path), invalid_clinical_report_qc=[])

    raw = raw.collect()
    assert excluded.collect().height == 0
    assert set(raw['clinicalReportId']) == {'R1', 'R2', 'R6', 'R7', 'R8'}
    assert raw.height == 12


def test_generate_is_independent_of_input_row_order(tmp_path: Path) -> None:
    raw, _ = generate(**_sources(tmp_path, REPORTS[::-1]), invalid_clinical_report_qc=INVALID_QC)

    assert _sorted(raw.collect().to_dicts()) == EXPECTED


def test_score_multiplies_stage_by_stop_reason_minimum() -> None:
    df = pl.DataFrame({
        'clinicalStage': ['PHASE_2_3', 'PRECLINICAL', 'PHASE_1', 'PHASE_2_3', 'NOT_A_STAGE'],
        'trialStopReasonCategories': [['Success', 'Negative'], [], ['unknown_reason'], None, None],
    })
    # 0.5*0.5, 0.01*null, 0.1*null, 0.5*1.0, null*1.0; the first four measured against spark
    got = df.with_columns(CLINICAL_PRECEDENCE.score.alias('s'))['s'].to_list()
    assert got[0] == pytest.approx(0.25)
    assert got[1] is None
    assert got[2] is None
    # a null trialStopReasonCategories takes the `IS NULL -> 1.0` branch, unlike an empty list
    assert got[3] == pytest.approx(0.5)
    assert got[4] is None


def test_direction_on_trait_is_protect_when_the_disease_is_present() -> None:
    df = pl.DataFrame({'diseaseId': ['MONDO:1', None]})
    assert df.select(DIRECTION_ON_TRAIT.alias('d'))['d'].to_list() == ['protect', None]


def test_direction_on_target_maps_action_types() -> None:
    df = pl.DataFrame({'actionType': ['INHIBITOR', 'AGONIST', 'UNKNOWN', None, 'inhibitor']})
    assert df.select(DIRECTION_ON_TARGET.alias('d'))['d'].to_list() == ['LoF', 'GoF', None, None, None]


def test_clinical_precedence_unique_fields_are_the_config_ones() -> None:
    assert CLINICAL_PRECEDENCE.datasource_id == 'clinical_precedence'
    assert CLINICAL_PRECEDENCE.unique_fields == [
        'targetId',
        'targetFromSourceId',
        'diseaseId',
        'datasourceId',
        'drugId',
        'clinicalReportId',
    ]


def test_clinical_precedence_step_scores_assigns_directions_and_writes_excluded_reports(tmp_path: Path) -> None:
    src = tmp_path / 'src'
    sources = _sources(src)
    dst = tmp_path / 'out'

    clinical_precedence(
        {
            **sources,
            'disease': write_disease(src / 'disease', ['EFO_1', 'EFO_2']),
            'target': write_target(src / 'target', ['ENSG1', 'ENSG2', 'ENSG3']),
            'literature': write_literature(src / 'literature', {'111': '2002-01-01'}),
        },
        {
            'evidence': str(dst / 'valid'),
            'failed_evidence': str(dst / 'failed'),
            'excluded_report': str(dst / 'excluded_report'),
        },
        {'invalid_clinical_report_qc': INVALID_QC},
        None,
    )

    valid = pl.read_parquet(dst / 'valid').sort('clinicalReportId', 'targetId')
    failed = pl.read_parquet(dst / 'failed')
    excluded = pl.read_parquet(dst / 'excluded_report')
    assert valid.select('clinicalReportId', 'targetId', 'score', 'directionOnTrait', 'directionOnTarget').rows() == [
        ('R1', 'ENSG1', 0.7, 'protect', 'LoF'),
        ('R1', 'ENSG2', 0.7, 'protect', 'LoF'),
        ('R1', 'ENSG3', 0.7, 'protect', 'GoF'),
        ('R2', 'ENSG1', 0.2, 'protect', 'LoF'),
        ('R2', 'ENSG2', 0.2, 'protect', 'LoF'),
    ]
    assert valid['publicationDate'].to_list()[:3] == ['2002-01-01'] * 3
    assert failed.select('clinicalReportId', 'targetFromSourceId').rows() == [('R8', 'ENSG5')]
    assert sorted(excluded['id']) == ['R6', 'R7']


def test_an_existing_evidence_destination_leaves_no_excluded_report(tmp_path: Path) -> None:
    src = tmp_path / 'src'
    dst = tmp_path / 'out'
    (dst / 'valid').mkdir(parents=True)
    with pytest.raises(ValueError, match='valid'):
        clinical_precedence(
            {
                **_sources(src),
                'disease': write_disease(src / 'disease', ['EFO_1', 'EFO_2']),
                'target': write_target(src / 'target', ['ENSG1', 'ENSG2', 'ENSG3']),
                'literature': write_literature(src / 'literature', {'111': '2002-01-01'}),
            },
            {
                'evidence': str(dst / 'valid'),
                'failed_evidence': str(dst / 'failed'),
                'excluded_report': str(dst / 'excluded_report'),
            },
            {'invalid_clinical_report_qc': INVALID_QC},
            None,
        )
    assert not (dst / 'excluded_report').exists()
