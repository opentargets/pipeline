"""Tests for evidence_epmc step."""

import pytest
from pyspark.sql import Row


def _make_cooc_row(
    pmid='12345',
    pmcid='PMC001',
    section='abstract',
    type='GP-DS',
    isMapped=True,  # noqa: N803
    text='short text',
    label1='BRCA1',
    keywordId1='ENSG001',  # noqa: N803
    keywordId2='EFO_0000311',  # noqa: N803
    start1=0,
    end1=5,
    start2=10,
    end2=15,
    evidence_score=0.8,
    year=2020,
):
    return Row(
        pmid=pmid,
        pmcid=pmcid,
        section=section,
        type=type,
        isMapped=isMapped,
        text=text,
        label1=label1,
        keywordId1=keywordId1,
        keywordId2=keywordId2,
        start1=start1,
        end1=end1,
        start2=start2,
        end2=end2,
        evidence_score=evidence_score,
        year=year,
    )


class TestEpmcFiltering:
    """Test the filtering logic of _compute_evidence."""

    def test_filters_non_gp_ds_type(self, spark):
        from pts.pyspark.evidence_epmc import _compute_evidence

        data = [
            _make_cooc_row(type='GP-DS', evidence_score=1.5),
            _make_cooc_row(type='GP-DS', pmid='99999', keywordId1='ENSG002', evidence_score=1.5),
            _make_cooc_row(type='DS-CD'),
        ]
        df = spark.createDataFrame(data)
        result = _compute_evidence(df)
        types = {r['targetFromSourceId'] for r in result.collect()}
        assert 'ENSG001' in types

    def test_filters_excluded_terms(self, spark):
        from pts.pyspark.evidence_epmc import _compute_evidence

        data = [
            _make_cooc_row(label1='TEC', evidence_score=1.5),
            _make_cooc_row(label1='BRCA1', pmid='99999', keywordId1='ENSG002', evidence_score=1.5),
            _make_cooc_row(label1='BRCA1', pmid='99998', keywordId1='ENSG003', evidence_score=1.5),
        ]
        df = spark.createDataFrame(data)
        result = _compute_evidence(df)
        targets = {r['targetFromSourceId'] for r in result.collect()}
        assert 'ENSG001' not in targets

    def test_filters_long_text(self, spark):
        from pts.pyspark.evidence_epmc import _compute_evidence

        long_text = 'x' * 600
        data = [
            _make_cooc_row(text=long_text, evidence_score=1.5),
            _make_cooc_row(text='short', pmid='99999', keywordId1='ENSG002', evidence_score=1.5),
            _make_cooc_row(text='short2', pmid='99998', keywordId1='ENSG003', evidence_score=1.5),
        ]
        df = spark.createDataFrame(data)
        result = _compute_evidence(df)
        assert result.count() == 2

    def test_filters_unmapped(self, spark):
        from pts.pyspark.evidence_epmc import _compute_evidence

        data = [
            _make_cooc_row(isMapped=False, evidence_score=1.5),
            _make_cooc_row(isMapped=True, pmid='99999', keywordId1='ENSG002', evidence_score=1.5),
            _make_cooc_row(isMapped=True, pmid='99998', keywordId1='ENSG003', evidence_score=1.5),
        ]
        df = spark.createDataFrame(data)
        result = _compute_evidence(df)
        assert result.count() == 2

    def test_resource_score_threshold(self, spark):
        """Only rows with resourceScore > 1 should pass."""
        from pts.pyspark.evidence_epmc import _compute_evidence

        data = [
            _make_cooc_row(evidence_score=0.5),
            _make_cooc_row(evidence_score=0.3, text='other'),
        ]
        df = spark.createDataFrame(data)
        result = _compute_evidence(df)
        assert result.count() == 0


class TestEpmcAggregation:
    """Test aggregation and output schema."""

    def test_aggregates_by_publication(self, spark):
        from pts.pyspark.evidence_epmc import _compute_evidence

        data = [
            _make_cooc_row(section='title', evidence_score=1.0, text='sentence 1'),
            _make_cooc_row(section='abstract', evidence_score=1.0, text='sentence 2'),
        ]
        df = spark.createDataFrame(data)
        result = _compute_evidence(df)
        assert result.count() == 1
        row = result.collect()[0]
        assert row['resourceScore'] == pytest.approx(2.0)

    def test_output_has_required_columns(self, spark):
        from pts.pyspark.evidence_epmc import _compute_evidence

        data = [
            _make_cooc_row(evidence_score=1.5),
            _make_cooc_row(evidence_score=1.0, text='another'),
        ]
        df = spark.createDataFrame(data)
        result = _compute_evidence(df)
        expected = {
            'datasourceId',
            'datatypeId',
            'targetFromSourceId',
            'diseaseFromSourceMappedId',
            'resourceScore',
            'literature',
            'textMiningSentences',
            'pmcIds',
            'publicationYear',
        }
        assert expected.issubset(set(result.columns))

    def test_text_mining_sentences_struct(self, spark):
        from pts.pyspark.evidence_epmc import _compute_evidence

        data = [
            _make_cooc_row(evidence_score=1.5, start1=0, end1=5, start2=10, end2=15),
            _make_cooc_row(evidence_score=1.0, start1=20, end1=25, start2=30, end2=35, text='s2'),
        ]
        df = spark.createDataFrame(data)
        result = _compute_evidence(df)
        row = result.collect()[0]
        sentences = row['textMiningSentences']
        assert len(sentences) == 2
        s = sentences[0]
        assert 'text' in s.asDict()
        assert 'tStart' in s.asDict()
        assert 'dEnd' in s.asDict()
