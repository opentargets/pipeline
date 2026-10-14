"""Tests for OT CRISPR (ot_crispr) evidence."""

from __future__ import annotations

import datetime
from pathlib import Path

import polars as pl
import pytest

from pts.transformers.evidence.ot_crispr import (
    OT_CRISPR,
    adjust_column_names,
    generate,
    ot_crispr,
    process_studies,
)
from tests.evidence.fixtures import write_disease, write_target

STUDY_HEADER = [
    'studyId',
    'projectId',
    'projectDescription',
    'releaseVersion',
    'releaseDate',
    'studyOverview',
    'diseases',
    'contrast',
    'crisprScreenLibrary',
    'crisprStudyMode',
    'cellType',
    'cellLineBackground',
    'isCellTypeDerived',
    'geneticBackground',
    'dataFileType',
    'filterColumn',
    'threshold',
    'replicateNumber',
    'dataFile',
    'ControlDataset',
]
DESCRIPTION_ROW = {c: f'Description of {c}' for c in STUDY_HEADER} | {'projectId': 'OpenTargets project identifier'}
MAGECK_COLUMNS = ['id', 'num', 'neg|score', 'neg|p-value', 'neg|fdr', 'neg|lfc', 'pos|score', 'pos|p-value', 'pos|fdr']
MAGECK_COLUMNS += ['pos|lfc']

RAW_SCHEMA = {
    'projectId': pl.String,
    'projectDescription': pl.String,
    'releaseDate': pl.Date,
    'releaseVersion': pl.String,
    'studyId': pl.String,
    'studyOverview': pl.String,
    'contrast': pl.String,
    'crisprScreenLibrary': pl.String,
    'cellType': pl.String,
    'cellLineBackground': pl.String,
    'diseaseFromSourceMappedId': pl.String,
    'targetFromSourceId': pl.String,
    'log2FoldChangeValue': pl.Float32,
    'resourceScore': pl.Float64,
    'statisticalTestTail': pl.String,
    'datasourceId': pl.String,
    'datatypeId': pl.String,
}


def _study(study_id: str | None, **fields: str | None) -> dict[str, str | None]:
    row: dict[str, str | None] = dict.fromkeys(STUDY_HEADER)
    row |= {
        'studyId': study_id,
        'projectId': 'OTAR001',
        'projectDescription': 'A project',
        'releaseVersion': 'OTAR001-1',
        'releaseDate': '2021-06-10',
        'studyOverview': f'Overview of {study_id}',
        'diseases': 'EFO_1',
        'contrast': 'A vs B',
        'cellType': 'neurons',
        'dataFileType': 'MAGeCK',
        'filterColumn': 'pos|p-value',
        'threshold': '0.05',
        'dataFile': 'a.txt',
    }
    return row | fields


def _write_study_table(path: Path, rows: list[dict[str, str | None]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame([DESCRIPTION_ROW, *rows], schema=dict.fromkeys(STUDY_HEADER, pl.String)).write_csv(path)
    return str(path)


def _write_mageck(path: Path, rows: list[dict[str, str]], separator: str = '|') -> None:
    """Write a MAGeCK gene summary; unset values are '1', so they never pass a threshold."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [c.replace('|', separator) for c in MAGECK_COLUMNS]
    lines = ['\t'.join(header)] + ['\t'.join(row.get(c, '1') for c in MAGECK_COLUMNS) for row in rows]
    path.write_text('\n'.join(lines) + '\n')


def _sources(src: Path) -> dict[str, str]:
    data = src / 'data'
    # S1: two replicates, both tails filtered; replicate 2 uses `.` as the label separator.
    _write_mageck(
        data / 'OTAR001' / 'rep1.txt',
        [
            {'id': 'GA_1', 'neg|p-value': '0.01', 'neg|lfc': '-1.5'},
            {'id': 'GB', 'pos|p-value': '0.05', 'pos|lfc': '2.0'},
            {'id': 'GC', 'neg|p-value': '0.2', 'pos|p-value': '0.3'},
            {'id': 'GD', 'neg|p-value': '0.001'},
            {'id': 'GE', 'pos|p-value': '0.03', 'pos|lfc': '1.0'},
            {'id': 'GF_1', 'neg|p-value': '0.001'},
            {'id': 'GF_2', 'neg|p-value': '0.002'},
        ],
    )
    _write_mageck(
        data / 'OTAR001' / 'rep2.txt',
        [
            {'id': 'GA', 'neg|p-value': '0.02', 'neg|lfc': '-0.5'},
            {'id': 'GB', 'pos|p-value': '0.04', 'pos|lfc': '1.0'},
            {'id': 'GC', 'neg|p-value': '0.001'},
            {'id': 'GE', 'pos|p-value': '0.005', 'pos|lfc': '3.0'},
        ],
        separator='.',
    )
    # S2: hits that are also hits in the control are removed.
    _write_mageck(
        data / 'OTAR001' / 'case.txt',
        [
            {'id': 'GX', 'pos|fdr': '0.001', 'pos|lfc': '3.0'},
            {'id': 'GY', 'pos|fdr': '0.005', 'pos|lfc': '4.0'},
            {'id': 'GZ', 'pos|fdr': '0.01', 'pos|lfc': '5.0'},
        ],
    )
    _write_mageck(
        data / 'OTAR001' / 'control.txt',
        [{'id': 'GX', 'pos|fdr': '0.001'}, {'id': 'GY', 'pos|fdr': '0.5'}],
    )
    two_tails = {'filterColumn': 'neg|p-value,pos|p-value', 'diseases': ' EFO_1|EFO_2|EFO_1'}
    study_table = _write_study_table(
        src / 'config.csv',
        [
            _study('S1', **two_tails, replicateNumber='1', dataFile='rep1.txt'),
            _study('S1', **two_tails, replicateNumber='2', dataFile='rep2.txt'),
            _study('S2', filterColumn='pos|fdr', threshold='0.01', dataFile='case.txt', ControlDataset='control.txt'),
            _study('S3', projectId='XYZ001'),
            _study('S4', diseases=None),
            _study('S5', dataFile=None),
            _study(None),
        ],
    )
    return {'study_table': study_table, 'ot_crispr_data': str(data)}


def test_process_studies_drops_invalid_studies_and_collects_replicates(tmp_path: Path) -> None:
    study_table = pl.read_csv(_sources(tmp_path)['study_table'], infer_schema=False)

    studies = process_studies(study_table).sort('studyId')

    assert studies['studyId'].to_list() == ['S1', 'S2']
    assert studies['diseaseFromSourceMappedId'].to_list() == [['EFO_1', 'EFO_2'], ['EFO_1']]
    assert studies['filterColumns'].to_list() == [['neg|p-value', 'pos|p-value'], ['pos|fdr']]
    assert studies['threshold'].dtype == pl.Float64
    assert studies['releaseDate'].to_list() == [datetime.date(2021, 6, 10)] * 2
    assert studies['replicates'].to_list() == [
        [
            {'dataFile': 'rep1.txt', 'ControlDataset': None, 'replicateId': '1'},
            {'dataFile': 'rep2.txt', 'ControlDataset': None, 'replicateId': '2'},
        ],
        [{'dataFile': 'case.txt', 'ControlDataset': 'control.txt', 'replicateId': '1'}],
    ]


def test_split_keeps_whitespace_after_a_separator(tmp_path: Path) -> None:
    """Only leading whitespace is removed, as spark's `regexp_replace(col, '^\\s+', '')`."""
    path = _write_study_table(tmp_path / 'config.csv', [_study('S1', diseases='\t EFO_1| EFO_2 ')])

    studies = process_studies(pl.read_csv(path, infer_schema=False))

    assert studies['diseaseFromSourceMappedId'].to_list() == [['EFO_1', ' EFO_2 ']]


def test_adjust_column_names_normalises_dots_and_rejects_unknown_separators() -> None:
    dotted = pl.DataFrame(schema=['id', 'num', 'neg.score', 'neg.p-value', 'pos.lfc'])
    assert adjust_column_names(dotted).columns == ['id', 'num', 'neg|score', 'neg|p-value', 'pos|lfc']

    with pytest.raises(ValueError, match='Unrecognized label separator in neg_p-value'):
        adjust_column_names(pl.DataFrame(schema=['id', 'num', 'neg_score', 'neg_p-value']))


def test_generate_keeps_hits_of_every_replicate(tmp_path: Path) -> None:
    """Hits are genes below the threshold in every replicate, with the best replicate's values.

    - S1 keeps GA and GE, found in both replicates, each with its smaller p-value's replicate;
    - GB's p-value of exactly 0.05 fails the 0.05 threshold, so it is not found in replicate 1;
    - GF has two ids in replicate 1 but is missing from replicate 2, so it is dropped;
    - S2 loses GX, a hit in its control too;
    - S3 to S5, the description row and the row without a studyId are dropped, so the null
      `diseases` of S4 never reaches the explode.
    """
    raw = generate(**_sources(tmp_path))

    assert raw.collect_schema() == pl.Schema(RAW_SCHEMA)
    out = raw.collect().sort('studyId', 'targetFromSourceId', 'diseaseFromSourceMappedId')
    assert out.select(
        'studyId', 'targetFromSourceId', 'diseaseFromSourceMappedId', 'resourceScore', 'statisticalTestTail'
    ).rows() == [
        ('S1', 'GA', 'EFO_1', 0.01, 'lower tail'),
        ('S1', 'GA', 'EFO_2', 0.01, 'lower tail'),
        ('S1', 'GE', 'EFO_1', 0.005, 'upper tail'),
        ('S1', 'GE', 'EFO_2', 0.005, 'upper tail'),
        ('S2', 'GY', 'EFO_1', 0.005, 'upper tail'),
    ]
    assert out['log2FoldChangeValue'].to_list() == [-1.5, -1.5, 3.0, 3.0, 4.0]
    assert out.select('projectId', 'releaseDate', 'datasourceId', 'datatypeId').unique().rows() == [
        ('OTAR001', datetime.date(2021, 6, 10), 'ot_crispr', 'ot_partner')
    ]


def test_generate_takes_the_first_matching_filter_column_on_a_tie(tmp_path: Path) -> None:
    _write_mageck(
        tmp_path / 'data' / 'OTAR001' / 'a.txt',
        [{'id': 'G1', 'neg|p-value': '0.001', 'pos|p-value': '0.001', 'neg|lfc': '-1.0', 'pos|lfc': '1.0'}],
    )
    study_table = _write_study_table(tmp_path / 'config.csv', [_study('S1', filterColumn='pos|p-value,neg|p-value')])

    out = generate(study_table, str(tmp_path / 'data')).collect()

    assert out.select('statisticalTestTail', 'log2FoldChangeValue').rows() == [('upper tail', 1.0)]


def test_generate_reads_nan_and_unparseable_values_as_spark_does(tmp_path: Path) -> None:
    """Unparseable filter and lfc values are null; NaN sorts after every number, so it never wins the min."""
    _write_mageck(
        tmp_path / 'data' / 'OTAR001' / 'a.txt',
        [
            {'id': 'G1', 'neg|p-value': 'NaN', 'pos|p-value': '0.001', 'pos|lfc': '1.0'},
            {'id': 'G2', 'neg|p-value': 'NA', 'pos|p-value': '0.002', 'pos|lfc': 'NA'},
            {'id': 'G3', 'neg|p-value': 'junk', 'pos|p-value': 'NaN'},
            {'id': 'G4', 'neg|p-value': '0.003', 'pos|p-value': 'junk', 'neg|lfc': 'junk'},
        ],
    )
    study_table = _write_study_table(tmp_path / 'config.csv', [_study('S1', filterColumn='neg|p-value,pos|p-value')])

    out = generate(study_table, str(tmp_path / 'data')).collect().sort('targetFromSourceId')

    assert out.select('targetFromSourceId', 'resourceScore', 'statisticalTestTail', 'log2FoldChangeValue').rows() == [
        ('G1', 0.001, 'upper tail', 1.0),
        ('G2', 0.002, 'upper tail', None),
        ('G4', 0.003, 'lower tail', None),
    ]


@pytest.mark.parametrize(
    ('threshold', 'expected'),
    [
        pytest.param('junk', [], id='unparseable threshold is null, so nothing passes'),
        pytest.param('NaN', ['G1', 'G2'], id='NaN threshold sorts last, so every number passes'),
    ],
)
def test_generate_reads_an_unusable_threshold_as_spark_does(tmp_path: Path, threshold: str, expected: list) -> None:
    _write_mageck(
        tmp_path / 'data' / 'OTAR001' / 'a.txt',
        [{'id': 'G1', 'pos|p-value': '0.5'}, {'id': 'G2', 'pos|p-value': '7'}, {'id': 'G3', 'pos|p-value': 'NaN'}],
    )
    study_table = _write_study_table(tmp_path / 'config.csv', [_study('S1', threshold=threshold)])

    out = generate(study_table, str(tmp_path / 'data')).collect()

    assert sorted(out['targetFromSourceId'].to_list()) == expected


def test_score_is_one() -> None:
    assert pl.select(OT_CRISPR.score.alias('s'))['s'].to_list() == [1.0]


def test_ot_crispr_step_writes_scored_evidence(tmp_path: Path) -> None:
    src = tmp_path / 'src'
    dst = tmp_path / 'out'

    ot_crispr(
        {
            **_sources(src),
            'disease': write_disease(src / 'disease', ['EFO_1']),
            'target': write_target(src / 'target', ['ENSG1', 'ENSG2'], {'ENSG1': 'GA', 'ENSG2': 'GE'}),
        },
        {'evidence': str(dst / 'valid'), 'failed_evidence': str(dst / 'failed')},
        {},
        None,
    )

    valid = pl.read_parquet(dst / 'valid').sort('targetId')
    assert valid.select('targetId', 'diseaseId', 'studyId', 'score').rows() == [
        ('ENSG1', 'EFO_1', 'S1', 1.0),
        ('ENSG2', 'EFO_1', 'S1', 1.0),
    ]
    assert pl.read_parquet(dst / 'failed').height == 3
