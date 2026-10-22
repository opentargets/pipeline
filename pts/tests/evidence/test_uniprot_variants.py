"""Tests for UniProt variants evidence."""

from __future__ import annotations

import polars as pl

from pts.transformers.evidence.uniprot_variants import UNIPROT_VARIANTS


def test_score_is_the_confidence_score() -> None:
    df = pl.DataFrame({'confidence': ['high', 'medium', 'low', None]})
    assert df.select(UNIPROT_VARIANTS.score.alias('s'))['s'].to_list() == [1.0, 0.5, None, None]


