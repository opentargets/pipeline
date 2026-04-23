"""Tests for literature_entity_lut step."""

import pytest
from pyspark.sql import Row
from pyspark.sql import functions as f
from pyspark.sql import types as t

_MATCH_SCHEMA = t.StructType([
    t.StructField('pmid', t.StringType()),
    t.StructField('pmcid', t.StringType()),
    t.StructField('date', t.StringType()),
    t.StructField('year', t.IntegerType()),
    t.StructField('month', t.IntegerType()),
    t.StructField('day', t.IntegerType()),
    t.StructField('keywordId', t.StringType()),
    t.StructField('type', t.StringType()),
    t.StructField('section', t.StringType()),
    t.StructField('isMapped', t.BooleanType()),
    t.StructField('organisms', t.StringType()),
    t.StructField('pubDate', t.StringType()),
    t.StructField('text', t.StringType()),
    t.StructField('trace_source', t.StringType()),
    t.StructField('labelN', t.StringType()),
])


class TestHarmonicFn:
    """Test the _harmonic_fn column expression."""

    def test_single_weight(self, spark):
        from pts.pyspark.literature_entity_lut import _harmonic_fn

        df = spark.createDataFrame(
            [([1.0],)],
            schema='weights ARRAY<DOUBLE>',
        )
        result = df.select(_harmonic_fn(f.col('weights'), f.size(f.col('weights'))).alias('score')).collect()[0][
            'score'
        ]
        # 1.0 / 1^2 = 1.0
        assert result == pytest.approx(1.0)

    def test_multiple_weights(self, spark):
        from pts.pyspark.literature_entity_lut import _harmonic_fn

        df = spark.createDataFrame(
            [([1.0, 0.8, 0.6],)],
            schema='weights ARRAY<DOUBLE>',
        )
        result = df.select(_harmonic_fn(f.col('weights'), f.size(f.col('weights'))).alias('score')).collect()[0][
            'score'
        ]
        # 1.0/1 + 0.8/4 + 0.6/9 = 1.0 + 0.2 + 0.0667 = 1.2667
        assert result == pytest.approx(1.0 + 0.2 + 0.6 / 9)

    def test_empty_weights(self, spark):
        from pts.pyspark.literature_entity_lut import _harmonic_fn

        df = spark.createDataFrame(
            [([],)],
            schema='weights ARRAY<DOUBLE>',
        )
        result = df.select(_harmonic_fn(f.col('weights'), f.size(f.col('weights'))).alias('score')).collect()[0][
            'score'
        ]
        assert result == pytest.approx(0.0)


class TestLiteratureEntityLutTransform:
    """Test the core transformation logic."""

    def test_title_gets_single_weight(self, spark):
        """Title section should use a single titleWeight, not collect_list."""
        from pts.pyspark.literature_entity_lut import _compute_relevance

        data = [
            Row(
                pmid='1',
                pmcid='PMC1',
                date='2020-01-01',
                year=2020,
                month=1,
                day=1,
                keywordId='ENSG001',
                type='GP',
                section='title',
                isMapped=True,
                organisms=None,
                pubDate=None,
                text='test',
                trace_source=None,
                labelN=None,
            ),
            Row(
                pmid='1',
                pmcid='PMC1',
                date='2020-01-01',
                year=2020,
                month=1,
                day=1,
                keywordId='ENSG001',
                type='GP',
                section='title',
                isMapped=True,
                organisms=None,
                pubDate=None,
                text='test2',
                trace_source=None,
                labelN=None,
            ),
        ]
        df = spark.createDataFrame(data, schema=_MATCH_SCHEMA)
        result = _compute_relevance(df)
        row = result.filter(f.col('keywordId') == 'ENSG001').collect()[0]
        # Two title occurrences should still yield a single [1.0], score = 1.0
        assert row['relevance'] == pytest.approx(1.0)

    def test_multi_section_scoring(self, spark):
        """Keyword appearing in title and abstract gets harmonic score across both."""
        from pts.pyspark.literature_entity_lut import _compute_relevance

        data = [
            Row(
                pmid='1',
                pmcid='PMC1',
                date='2020-01-01',
                year=2020,
                month=1,
                day=1,
                keywordId='ENSG001',
                type='GP',
                section='title',
                isMapped=True,
                organisms=None,
                pubDate=None,
                text='t1',
                trace_source=None,
                labelN=None,
            ),
            Row(
                pmid='1',
                pmcid='PMC1',
                date='2020-01-01',
                year=2020,
                month=1,
                day=1,
                keywordId='ENSG001',
                type='GP',
                section='abstract',
                isMapped=True,
                organisms=None,
                pubDate=None,
                text='t2',
                trace_source=None,
                labelN=None,
            ),
        ]
        df = spark.createDataFrame(data, schema=_MATCH_SCHEMA)
        result = _compute_relevance(df)
        row = result.collect()[0]
        # title weight=1.0, abstract weight=0.8
        # harmonic: 1.0/1 + 0.8/4 = 1.2
        assert row['relevance'] == pytest.approx(1.2)

    def test_output_columns(self, spark):
        """Output should contain exactly the expected columns."""
        from pts.pyspark.literature_entity_lut import _compute_relevance

        data = [
            Row(
                pmid='1',
                pmcid='PMC1',
                date='2020-01-01',
                year=2020,
                month=1,
                day=1,
                keywordId='ENSG001',
                type='GP',
                section='title',
                isMapped=True,
                organisms=None,
                pubDate=None,
                text='t',
                trace_source=None,
                labelN=None,
            ),
        ]
        df = spark.createDataFrame(data, schema=_MATCH_SCHEMA)
        result = _compute_relevance(df)
        expected = {'pmid', 'pmcid', 'date', 'year', 'month', 'day', 'keywordId', 'relevance', 'keywordType'}
        assert set(result.columns) == expected

    def test_unmapped_section_gets_default_weight(self, spark):
        """A section not in the rank table gets rank=100, weight=0.01."""
        from pts.pyspark.literature_entity_lut import _compute_relevance

        data = [
            Row(
                pmid='1',
                pmcid='PMC1',
                date='2020-01-01',
                year=2020,
                month=1,
                day=1,
                keywordId='ENSG001',
                type='GP',
                section='unknown_section',
                isMapped=True,
                organisms=None,
                pubDate=None,
                text='t',
                trace_source=None,
                labelN=None,
            ),
        ]
        df = spark.createDataFrame(data, schema=_MATCH_SCHEMA)
        result = _compute_relevance(df)
        row = result.collect()[0]
        # 0.01 / 1^2 = 0.01
        assert row['relevance'] == pytest.approx(0.01)
