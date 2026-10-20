"""Tests for ENCORE evidence."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from pts.transformers.evidence.encore import ENCORE, encore
from tests.evidence.fixtures import write_disease, write_target


def test_score_is_the_absolute_interaction_score_over_16() -> None:
    df = pl.DataFrame({'geneticInteractionScore': [-32.0, 8.0, 0.0, None]})
    assert df.select(ENCORE.score)['geneticInteractionScore'].to_list() == [2.0, 0.5, 0.0, None]


def test_cell_lines_tell_evidence_apart(tmp_path: Path) -> None:
    src = tmp_path / 'src'
    (src / 'encore').mkdir(parents=True)
    pl.DataFrame({
        'datasourceId': ['encore'] * 3,
        'targetFromSourceId': ['A1'] * 3,
        'interactingTargetFromSourceId': ['B1'] * 3,
        'diseaseFromSourceMappedId': ['EFO_1'] * 3,
        'diseaseCellLines': [[{'id': 'C1'}], [{'id': 'C2'}], [{'id': 'C2'}]],
        'geneticInteractionScore': [-8.0, 4.0, 4.0],
    }).write_parquet(src / 'encore' / 'part.parquet')
    dst = tmp_path / 'out'

    encore(
        {
            'evidence': str(src / 'encore'),
            'disease': write_disease(src / 'disease', ['EFO_1']),
            'target': write_target(src / 'target', ['ENSG1'], {'ENSG1': 'A1'}),
        },
        {'evidence': str(dst / 'valid'), 'failed_evidence': str(dst / 'failed')},
        {},
        None,
    )

    valid = pl.read_parquet(dst / 'valid')
    failed = pl.read_parquet(dst / 'failed')
    assert sorted(valid['score'].to_list()) == [0.25, 0.5]
    assert valid['targetId'].to_list() == ['ENSG1', 'ENSG1']
    assert failed.height == 1
