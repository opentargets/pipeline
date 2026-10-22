"""Tests for Reactome evidence."""

from __future__ import annotations

import polars as pl

from pts.transformers.evidence.reactome import REACTOME


def test_score_is_one() -> None:
    df = pl.DataFrame({'dummy': ['x', None]})
    assert df.with_columns(REACTOME.score.alias('s'))['s'].to_list() == [1.0, 1.0]


