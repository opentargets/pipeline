"""Tests for Reactome evidence."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from pts.transformers.evidence.reactome import REACTOME, reactome
from tests.evidence.fixtures import write_disease, write_literature, write_target


def test_score_is_one() -> None:
    df = pl.DataFrame({'dummy': ['x', None]})
    assert df.with_columns(REACTOME.score.alias('s'))['s'].to_list() == [1.0, 1.0]


def test_unknown_target_is_flagged_invalid(tmp_path: Path) -> None:
    src = tmp_path / 'src'
    src.mkdir()
    rows = [
        {
            'datasourceId': 'reactome',
            'targetFromSourceId': target,
            'diseaseFromSourceMappedId': 'EFO_1',
            'reactionId': 'R-1',
            'variantAminoacidDescriptions': ['p.A1B'],
            'targetModulation': 'up',
            'literature': ['1'],
        }
        for target in ('A1', 'ZZZ')
    ]
    (src / 'reactome.json').write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
    dst = tmp_path / 'out'

    reactome(
        {
            'evidence': str(src / 'reactome.json'),
            'disease': write_disease(src / 'disease', ['EFO_1']),
            'target': write_target(src / 'target', ['ENSG1'], {'ENSG1': 'A1'}),
            'literature': write_literature(src / 'literature', {'1': '2020-01-01'}),
        },
        {'evidence': str(dst / 'valid'), 'failed_evidence': str(dst / 'failed')},
        {},
        None,
    )

    valid = pl.read_parquet(dst / 'valid')
    failed = pl.read_parquet(dst / 'failed')
    assert valid['targetId'].to_list() == ['ENSG1']
    assert valid['score'].to_list() == [1.0]
    assert failed['targetFromSourceId'].to_list() == ['ZZZ']
