"""Tests for UniProt literature evidence."""

from __future__ import annotations

import polars as pl

from pts.transformers.evidence.uniprot_literature import UNIPROT_LITERATURE


def test_score_is_the_confidence_score() -> None:
    df = pl.DataFrame({'confidence': ['high', 'medium', 'low', None]})
    assert df.select(UNIPROT_LITERATURE.score.alias('s'))['s'].to_list() == [1.0, 0.5, None, None]


