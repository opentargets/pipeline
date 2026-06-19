"""Tests for literature_cooccurrence_evidence step."""

import pytest
from pyspark.sql import Row


class TestAdaptCooccurrenceForEvidence:
    """Test the Cooccurrence -> _compute_evidence column adapter."""

    def test_renames_columns(self, spark):
        from pts.pyspark.literature_cooccurrence_evidence import (
            _adapt_cooccurrence_for_evidence,
        )

        df = spark.createDataFrame([
            Row(
                mappedId1='ENSG001',
                mappedId2='EFO_0000311',
                evidenceScore=0.8,
                section='abstract',
            ),
        ])
        result = _adapt_cooccurrence_for_evidence(df)
        assert 'keywordId1' in result.columns
        assert 'keywordId2' in result.columns
        assert 'evidence_score' in result.columns

    def test_drops_old_names(self, spark):
        from pts.pyspark.literature_cooccurrence_evidence import (
            _adapt_cooccurrence_for_evidence,
        )

        df = spark.createDataFrame([
            Row(
                mappedId1='ENSG001',
                mappedId2='EFO_0000311',
                evidenceScore=0.8,
                section='abstract',
            ),
        ])
        result = _adapt_cooccurrence_for_evidence(df)
        assert 'mappedId1' not in result.columns
        assert 'mappedId2' not in result.columns
        assert 'evidenceScore' not in result.columns

    def test_preserves_other_columns_and_values(self, spark):
        from pts.pyspark.literature_cooccurrence_evidence import (
            _adapt_cooccurrence_for_evidence,
        )

        df = spark.createDataFrame([
            Row(
                mappedId1='ENSG001',
                mappedId2='EFO_0000311',
                evidenceScore=0.8,
                section='abstract',
            ),
        ])
        result = _adapt_cooccurrence_for_evidence(df)
        assert 'section' in result.columns
        row = result.collect()[0]
        assert row['keywordId1'] == 'ENSG001'
        assert row['evidence_score'] == pytest.approx(0.8)
