"""Tests for Expression Atlas evidence."""

from __future__ import annotations

from math import nan

import polars as pl
import pytest

from pts.transformers.evidence.expression_atlas import EXPRESSION_ATLAS


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


