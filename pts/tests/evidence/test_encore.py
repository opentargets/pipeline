"""Tests for ENCORE evidence."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from pts.schemas.outputs.encore_evidence import EncoreEvidence
from pts.transformers.evidence.encore import ENCORE, encore
from tests.evidence.fixtures import write_disease, write_target

#: Columns post-processing adds, absent from raw ENCORE evidence.
_DERIVED = {'diseaseId', 'targetId', 'id', 'qualityControls', 'score', 'evidenceDate'}


def test_score_is_the_absolute_interaction_score_over_16() -> None:
    df = pl.DataFrame({'geneticInteractionScore': [-32.0, 8.0, 0.0, None]})
    assert df.select(ENCORE.score)['geneticInteractionScore'].to_list() == [2.0, 0.5, 0.0, None]


def test_cell_lines_tell_evidence_apart(tmp_path: Path) -> None:
    src = tmp_path / 'src'
    (src / 'encore').mkdir(parents=True)
    # complete rows, so the written evidence passes `EncoreEvidence`
    raw = {k: v for k, v in EncoreEvidence.example.items() if k not in _DERIVED}
    cell_line = raw['diseaseCellLines'][0]
    pl.DataFrame([
        raw
        | {
            'targetFromSourceId': 'A1',
            'interactingTargetFromSourceId': 'B1',
            'diseaseFromSourceMappedId': 'EFO_1',
            'diseaseCellLines': [cell_line | {'id': cell_line_id}],
            'geneticInteractionScore': score,
        }
        for cell_line_id, score in [('C1', -8.0), ('C2', 4.0), ('C2', 4.0)]
    ]).write_parquet(src / 'encore' / 'part.parquet')
    dst = tmp_path / 'out'

    encore(
        {
            'evidence': str(src / 'encore'),
            'disease': write_disease(src / 'disease', ['EFO_1']),
            'target': write_target(src / 'target', ['ENSG00000000001'], {'ENSG00000000001': 'A1'}),
        },
        {'evidence': str(dst / 'valid'), 'failed_evidence': str(dst / 'failed')},
        {},
        None,
    )

    valid = pl.read_parquet(dst / 'valid')
    failed = pl.read_parquet(dst / 'failed')
    assert sorted(valid['score'].to_list()) == [0.25, 0.5]
    assert valid['targetId'].to_list() == ['ENSG00000000001', 'ENSG00000000001']
    assert failed.height == 1


def test_validation_readouts_are_harmonised_before_validation(tmp_path: Path) -> None:
    src = tmp_path / 'src'
    (src / 'encore').mkdir(parents=True)
    raw = {k: v for k, v in EncoreEvidence.example.items() if k not in _DERIVED}
    # fields reversed and `hsaValue` an integer, as a producer might export them
    readout = {'screen': 'combinatorial', 'readoutMethodName': 'CellTiter-Glo', 'isValidated': True, 'hsaValue': 1}
    pl.DataFrame([raw | {'targetFromSourceId': 'A1', 'validationReadouts': [readout]}]).write_parquet(
        src / 'encore' / 'part.parquet'
    )
    dst = tmp_path / 'out'

    encore(
        {
            'evidence': str(src / 'encore'),
            'disease': write_disease(src / 'disease', [raw['diseaseFromSourceMappedId']]),
            'target': write_target(src / 'target', ['ENSG00000000001'], {'ENSG00000000001': 'A1'}),
        },
        {'evidence': str(dst / 'valid'), 'failed_evidence': str(dst / 'failed')},
        {},
        None,
    )

    assert pl.read_parquet(dst / 'valid')['validationReadouts'].to_list() == [
        [{'hsaValue': 1.0, 'isValidated': True, 'readoutMethodName': 'CellTiter-Glo', 'screen': 'combinatorial'}]
    ]
