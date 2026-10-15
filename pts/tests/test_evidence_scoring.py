"""Tests for pts.transformers.evidence.scoring."""

import polars as pl

from pts.transformers.evidence.utils import flags
from pts.transformers.evidence.utils.scoring import calculate_evidence_score


class TestCalculateEvidenceScore:
    def test_copies_the_score_column(self) -> None:
        df = pl.DataFrame({'resourceScore': [0.5]})
        result = calculate_evidence_score(df, pl.col('resourceScore'))
        assert result.to_dicts()[0]['score'] == 0.5
        assert result.to_dicts()[0]['qualityControls'] == []

    def test_flags_null_negative_and_out_of_range_scores(self) -> None:
        df = pl.DataFrame({'resourceScore': [None, -0.1, 1.5, 1.0]}, schema={'resourceScore': pl.Float64})
        result = calculate_evidence_score(df, pl.col('resourceScore'))
        assert [r['qualityControls'] for r in result.to_dicts()] == [
            [flags.NO_VALID_SCORE],
            [flags.NO_VALID_SCORE],
            [flags.NO_VALID_SCORE],
            [],
        ]

    def test_evaluates_a_real_expression_not_just_a_column_copy(self) -> None:
        df = pl.DataFrame({'geneticInteractionScore': [-16.0, -8.0, 0.0, -32.0]})
        result = calculate_evidence_score(df, (-pl.col('geneticInteractionScore') / 16.0).clip(0.0, 1.0))
        assert result['score'].to_list() == [1.0, 0.5, 0.0, 1.0]
