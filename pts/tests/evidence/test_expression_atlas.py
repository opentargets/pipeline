"""Tests for Expression Atlas evidence."""

from __future__ import annotations

from math import nan
from pathlib import Path

import polars as pl
import pytest

from pts.transformers.evidence.expression_atlas import EXPRESSION_ATLAS, expression_atlas
from tests.evidence.fixtures import write_disease, write_literature, write_target


def test_score_rescales_the_p_value_by_fold_change_and_rank() -> None:
    df = pl.DataFrame({
        'resourceScore': [0.001, None, nan, 0.0],
        'log2FoldChangeValue': [-4.0, 2.0, 2.0, 1.0],
        'log2FoldChangePercentileRank': [80.0, 50.0, 50.0, 1.0],
    })
    got = df.select(EXPRESSION_ATLAS.score.alias('s'))['s'].to_list()
    assert got == pytest.approx([0.096, 1.0, 1.0, 0.001])


def test_score_is_one_for_null_or_nan_resource_score() -> None:
    df = pl.DataFrame({
        'resourceScore': [None, nan],
        'log2FoldChangeValue': [2.0, 2.0],
        'log2FoldChangePercentileRank': [50.0, 50.0],
    })
    got = df.select(EXPRESSION_ATLAS.score.alias('s'))['s'].to_list()
    assert got == pytest.approx([1.0, 1.0])


def test_excluded_biotype_is_flagged_invalid(tmp_path: Path) -> None:
    src = tmp_path / 'src'
    src.mkdir()
    pl.DataFrame({
        'datasourceId': ['expression_atlas'] * 2,
        'targetFromSourceId': ['A1', 'A2'],
        'diseaseFromSourceMappedId': ['EFO_1'] * 2,
        'contrast': ['c'] * 2,
        'studyId': ['s'] * 2,
        'resourceScore': [0.001] * 2,
        'log2FoldChangeValue': [4.0] * 2,
        'log2FoldChangePercentileRank': [80.0] * 2,
    }).write_parquet(src / 'atlas.parquet')
    target = write_target(src / 'target', ['ENSG1', 'ENSG2'], {'ENSG1': 'A1', 'ENSG2': 'A2'})
    pl.read_parquet(Path(target) / 'target.parquet').with_columns(
        pl.when(pl.col('id') == 'ENSG2').then(pl.lit('rRNA')).otherwise(pl.col('biotype')).alias('biotype')
    ).write_parquet(Path(target) / 'target.parquet')
    dst = tmp_path / 'out'

    expression_atlas(
        {
            'evidence': str(src / 'atlas.parquet'),
            'disease': write_disease(src / 'disease', ['EFO_1']),
            'target': target,
            'literature': write_literature(src / 'literature', {}),
        },
        {'evidence': str(dst / 'valid'), 'failed_evidence': str(dst / 'failed')},
        {},
        None,
    )

    valid = pl.read_parquet(dst / 'valid')
    failed = pl.read_parquet(dst / 'failed')
    assert valid['targetId'].to_list() == ['ENSG1']
    assert failed['targetId'].to_list() == ['ENSG2']
    assert 'Invalid biotype' in str(failed['qualityControls'].to_list())
