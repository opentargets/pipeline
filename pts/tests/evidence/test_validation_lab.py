"""Tests for Validation Lab (ot_crispr_validation) evidence."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from pts.transformers.evidence.validation_lab import OT_CRISPR_VALIDATION, validation_lab
from tests.evidence.fixtures import write_disease, write_target


def test_score_is_the_resource_score() -> None:
    df = pl.DataFrame({'resourceScore': [0.3, None]})
    assert df.select(OT_CRISPR_VALIDATION.score.alias('s'))['s'].to_list() == [0.3, None]


def test_evidence_is_scored_and_unknown_target_is_flagged(tmp_path: Path) -> None:
    src = tmp_path / 'src'
    src.mkdir()
    rows = [
        {
            'datasourceId': 'ot_crispr_validation',
            'targetFromSourceId': target,
            'diseaseFromSourceMappedId': 'EFO_1',
            'resourceScore': 0.3,
            'diseaseCellLines': [{'id': 'C1'}],
            'primaryProjectId': 'P1',
        }
        for target in ('A1', 'ZZZ')
    ]
    (src / 'validation_lab.json').write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
    dst = tmp_path / 'out'

    validation_lab(
        {
            'evidence': str(src / 'validation_lab.json'),
            'disease': write_disease(src / 'disease', ['EFO_1']),
            'target': write_target(src / 'target', ['ENSG1'], {'ENSG1': 'A1'}),
        },
        {'evidence': str(dst / 'valid'), 'failed_evidence': str(dst / 'failed')},
        {},
        None,
    )

    valid = pl.read_parquet(dst / 'valid')
    failed = pl.read_parquet(dst / 'failed')
    assert valid['targetId'].to_list() == ['ENSG1']
    assert valid['score'].to_list() == [0.3]
    assert failed['targetFromSourceId'].to_list() == ['ZZZ']
