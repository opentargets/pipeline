"""Tests for ClinVar and ClinVar somatic evidence."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import polars as pl
import pytest

from pts.transformers.evidence.clinvar import DIRECTION_ON_TARGET, DIRECTION_ON_TRAIT, SCORE, clinvar
from pts.transformers.evidence.clinvar_somatic import clinvar_somatic
from tests.evidence.fixtures import write_disease, write_literature, write_target


def test_score_sums_significance_and_confidence() -> None:
    df = pl.DataFrame({
        'clinicalSignificances': [['pathogenic'], [None], ['unknown_value'], None, ['likely pathogenic', 'pathogenic']],
        'confidence': [
            'practice guideline',
            'criteria provided, single submitter',
            'unknown_conf',
            None,
            'criteria provided, single submitter',
        ],
    })
    # the last row has two matching, non-equal-scoring elements: the best one counts, they do not add up
    assert df.select(SCORE)['clinicalSignificances'].to_list() == pytest.approx([1.0, 0.02, 0.0, 0.0, 0.92])


def test_direction_on_trait_null_when_both_pathogenic_and_protective() -> None:
    df = pl.DataFrame({
        'clinicalSignificances': [['pathogenic'], ['protective'], ['pathogenic', 'protective'], [None], ['Pathogenic']]
    })
    # the capitalised last row pins the case-insensitive match
    assert df.select(DIRECTION_ON_TRAIT.alias('d'))['d'].to_list() == ['risk', 'protect', None, None, 'risk']


def test_direction_on_target_is_lof_for_loss_of_function_consequences() -> None:
    df = pl.DataFrame({'variantFunctionalConsequenceId': ['SO_0001589', 'SO_0001893', 'SO_9999999', None]})
    assert df.select(DIRECTION_ON_TARGET.alias('d'))['d'].to_list() == ['LoF', 'LoF', None, None]


def _write_evidence(path: Path, rows: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, 'wt') as f:
        f.write('\n'.join(json.dumps(row) for row in rows) + '\n')
    return str(path)


def _source(tmp_path: Path, rows: list[dict]) -> dict[str, str]:
    return {
        'evidence': _write_evidence(tmp_path / 'src' / 'eva.json.gz', rows),
        'disease': write_disease(tmp_path / 'src' / 'disease', ['EFO_1']),
        'target': write_target(tmp_path / 'src' / 'target', ['ENSG1', 'ENSG2'], {'ENSG1': 'A1', 'ENSG2': 'A2'}),
        'literature': write_literature(tmp_path / 'src' / 'literature', {}),
    }


def _row(datasource: str, target: str, study: str, **extra: object) -> dict:
    return {
        'datasourceId': datasource,
        'targetFromSourceId': target,
        'diseaseFromSourceMappedId': 'EFO_1',
        'studyId': study,
        'variantId': '1_100_A_T',
        'clinicalSignificances': ['pathogenic'],
        'confidence': 'practice guideline',
        **extra,
    }


def _destination(tmp_path: Path, name: str) -> dict[str, str]:
    return {'evidence': str(tmp_path / name / 'valid'), 'failed_evidence': str(tmp_path / name / 'failed')}


def test_each_step_keeps_only_its_own_datasource(tmp_path: Path) -> None:
    source = _source(
        tmp_path,
        [
            _row('eva', 'A1', 'RCV1', variantFunctionalConsequenceId='SO_0001589'),
            _row('eva', 'A2', 'RCV2', clinicalSignificances=['protective']),
            _row('eva_somatic', 'A1', 'RCV3'),
        ],
    )
    germline, somatic = _destination(tmp_path, 'eva'), _destination(tmp_path, 'eva_somatic')

    clinvar(source, germline, {}, None)
    clinvar_somatic(source, somatic, {}, None)

    valid = pl.read_parquet(germline['evidence'])
    assert sorted(valid['studyId'].to_list()) == ['RCV1', 'RCV2']
    assert valid.sort('studyId')['directionOnTrait'].to_list() == ['risk', 'protect']
    assert pl.read_parquet(germline['failed_evidence']).height == 0
    assert pl.read_parquet(somatic['evidence'])['studyId'].to_list() == ['RCV3']
    assert pl.read_parquet(somatic['evidence'])['directionOnTrait'].to_list() == ['risk']


def test_a_column_first_appearing_late_is_still_read(tmp_path: Path) -> None:
    rows = [_row('eva', 'A1', f'RCV{i}') for i in range(150)]
    rows.append(_row('eva', 'A2', 'RCVLAST', variantFunctionalConsequenceId='SO_0001589'))
    dst = _destination(tmp_path, 'out')
    clinvar(_source(tmp_path, rows), dst, {}, None)

    last = pl.read_parquet(dst['evidence']).filter(pl.col('studyId') == 'RCVLAST')
    assert last['directionOnTarget'].to_list() == ['LoF']


def test_score_sum_is_exact_to_two_decimals() -> None:
    df = pl.DataFrame({
        'clinicalSignificances': [['pathogenic'], ['likely pathogenic']],
        'confidence': ['criteria provided, multiple submitters, no conflicts'] * 2,
    })
    # spark adds decimal literals, so 0.9 + 0.05 is 0.95 and not 0.9500000000000001
    assert df.select(SCORE)['clinicalSignificances'].to_list() == [0.95, 0.75]
