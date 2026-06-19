"""Tests for literature_publication_match step."""


class TestEpmcReadPath:
    """Test the EPMC jsonl glob path builder."""

    def test_with_date_prefix(self):
        from pts.pyspark.literature_publication_match import _epmc_read_path

        assert (
            _epmc_read_path('gs://otar025-epmc/ml02', 'fulltext', '2026_03')
            == 'gs://otar025-epmc/ml02/fulltext/2026_03*/*.jsonl'
        )

    def test_without_date_prefix(self):
        from pts.pyspark.literature_publication_match import _epmc_read_path

        assert (
            _epmc_read_path('gs://otar025-epmc/ml02', 'abstract', None) == 'gs://otar025-epmc/ml02/abstract/*/*.jsonl'
        )

    def test_strips_trailing_slash(self):
        from pts.pyspark.literature_publication_match import _epmc_read_path

        assert (
            _epmc_read_path('gs://otar025-epmc/ml02/', 'fulltext', None) == 'gs://otar025-epmc/ml02/fulltext/*/*.jsonl'
        )

    def test_empty_date_prefix_treated_as_none(self):
        from pts.pyspark.literature_publication_match import _epmc_read_path

        assert _epmc_read_path('gs://otar025-epmc/ml02', 'abstract', '') == 'gs://otar025-epmc/ml02/abstract/*/*.jsonl'
