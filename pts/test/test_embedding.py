"""Tests for literature_embedding step."""

from pyspark.sql import Row


class TestFilterMatches:
    """Test _filter_matches filtering logic."""

    def test_filters_by_type_and_mapping(self, spark):
        from pts.pyspark.literature_embedding import _filter_matches

        data = [
            Row(keywordId='ENSG001', type='GP', isMapped=True, section='title'),
            Row(keywordId='CHEMBL1', type='CD', isMapped=True, section='abstract'),
            Row(keywordId='EFO001', type='DS', isMapped=True, section='title'),
            Row(keywordId='ENSG002', type='GP', isMapped=False, section='title'),
            Row(keywordId='OTHER1', type='XX', isMapped=True, section='title'),
        ]
        df = spark.createDataFrame(data)
        result = _filter_matches(df)
        assert result.count() == 3
        ids = {r['keywordId'] for r in result.collect()}
        assert ids == {'ENSG001', 'CHEMBL1', 'EFO001'}

    def test_empty_input(self, spark):
        from pts.pyspark.literature_embedding import _filter_matches

        schema = 'keywordId STRING, type STRING, isMapped BOOLEAN, section STRING'
        df = spark.createDataFrame([], schema=schema)
        result = _filter_matches(df)
        assert result.count() == 0


class TestRegroupMatches:
    """Test _regroup_matches keyword collection and permutation logic."""

    def test_groups_by_pmid_and_section(self, spark):
        from pts.pyspark.literature_embedding import _regroup_matches

        data = [
            Row(pmid='1', keywordId='ENSG001', section='title', type='GP', isMapped=True),
            Row(pmid='1', keywordId='CHEMBL1', section='title', type='CD', isMapped=True),
            Row(pmid='1', keywordId='EFO001', section='abstract', type='DS', isMapped=True),
        ]
        df = spark.createDataFrame(data)
        result = _regroup_matches(df, max_sentence_length=100)
        assert result.count() > 0
        assert 'terms' in result.columns
        assert 'pmid' in result.columns
