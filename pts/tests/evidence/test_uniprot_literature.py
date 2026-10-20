"""Tests for UniProt literature evidence."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from pts.transformers.evidence.uniprot_literature import UNIPROT_LITERATURE, uniprot_literature
from tests.evidence.fixtures import write_disease, write_literature, write_target


def test_score_is_the_confidence_score() -> None:
    df = pl.DataFrame({'confidence': ['high', 'medium', 'low', None]})
    assert df.select(UNIPROT_LITERATURE.score.alias('s'))['s'].to_list() == [1.0, 0.5, None, None]


def test_evidence_is_scored_and_unknown_disease_is_flagged(tmp_path: Path) -> None:
    src = tmp_path / 'src'
    src.mkdir()
    rows = [
        {
            'datasourceId': 'uniprot_literature',
            'targetFromSourceId': 'A1',
            'diseaseFromSourceMappedId': disease,
            'diseaseFromSource': 'd',
            'confidence': 'medium',
            'literature': ['1'],
        }
        for disease in ('EFO_1', 'EFO_9')
    ]
    (src / 'uniprot_literature.json').write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
    dst = tmp_path / 'out'

    uniprot_literature(
        {
            'evidence': str(src / 'uniprot_literature.json'),
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
    assert valid['diseaseId'].to_list() == ['EFO_1']
    assert valid['score'].to_list() == [0.5]
    assert failed['diseaseFromSourceMappedId'].to_list() == ['EFO_9']
