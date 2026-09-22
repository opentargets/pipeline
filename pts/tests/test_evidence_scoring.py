"""Tests for pts.transformers.evidence.scoring."""

import polars as pl

from pts.transformers.evidence import flags
from pts.transformers.evidence.scoring import calculate_evidence_score


class TestCalculateEvidenceScore:
    def test_copies_the_score_column(self) -> None:
        df = pl.DataFrame({'resourceScore': [0.5]})
        result = calculate_evidence_score(df, 'resourceScore')
        assert result.to_dicts()[0]['score'] == 0.5
        assert result.to_dicts()[0]['qualityControls'] == []

    def test_flags_null_negative_and_out_of_range_scores(self) -> None:
        df = pl.DataFrame({'resourceScore': [None, -0.1, 1.5, 1.0]}, schema={'resourceScore': pl.Float64})
        result = calculate_evidence_score(df, 'resourceScore')
        assert [r['qualityControls'] for r in result.to_dicts()] == [
            [flags.NO_VALID_SCORE],
            [flags.NO_VALID_SCORE],
            [flags.NO_VALID_SCORE],
            [],
        ]
