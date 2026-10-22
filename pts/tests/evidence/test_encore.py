"""Tests for ENCORE evidence."""

from __future__ import annotations

import polars as pl

from pts.transformers.evidence.encore import ENCORE


def test_score_is_the_absolute_interaction_score_over_16() -> None:
    df = pl.DataFrame({'geneticInteractionScore': [-32.0, 8.0, 0.0, None]})
    assert df.select(ENCORE.score)['geneticInteractionScore'].to_list() == [2.0, 0.5, 0.0, None]




