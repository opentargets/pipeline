"""Tests for Validation Lab (ot_crispr_validation) evidence."""

from __future__ import annotations

import polars as pl

from pts.transformers.evidence.validation_lab import OT_CRISPR_VALIDATION


def test_score_is_the_resource_score() -> None:
    df = pl.DataFrame({'resourceScore': [0.3, None]})
    assert df.select(OT_CRISPR_VALIDATION.score.alias('s'))['s'].to_list() == [0.3, None]


