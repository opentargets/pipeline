"""Tests for the polars vep_view transformer."""

import polars as pl
import pytest

from pts.transformers.vep_view import process_credible_set, process_l2g


class TestProcessCredibleSet:
    """Tests for process_credible_set."""

    @pytest.fixture
    def credible_set(self) -> pl.LazyFrame:
        """Single locus with two tag variants; the lead is the first one."""
        return pl.from_dicts([
            {
                'studyLocusId': 'locus1',
                'studyId': 'study1',
                'variantId': '1_100_A_T',
                'finemappingMethod': 'SuSiE',
                'locus': [
                    {
                        'variantId': '1_100_A_T',
                        'pValueMantissa': 5.0,
                        'pValueExponent': -8,
                        'beta': 0.1,
                        'is95CredibleSet': True,
                        'is99CredibleSet': True,
                        'posteriorProbability': 0.8,
                    },
                    {
                        'variantId': '1_200_C_G',
                        'pValueMantissa': 1.0,
                        'pValueExponent': -5,
                        'beta': 0.05,
                        'is95CredibleSet': True,
                        'is99CredibleSet': True,
                        'posteriorProbability': 0.2,
                    },
                ],
            }
        ]).lazy()

    def test_lead_variant_is_flagged(self, credible_set: pl.LazyFrame) -> None:
        """The tag variant matching the locus lead variantId must have isLead=True."""
        result = process_credible_set(credible_set).collect()
        lead_rows = result.filter(pl.col('variantId') == '1_100_A_T')
        assert lead_rows['isLead'].to_list() == [True]

    def test_tag_variant_not_flagged(self, credible_set: pl.LazyFrame) -> None:
        """A tag variant that is not the lead must have isLead=False."""
        result = process_credible_set(credible_set).collect()
        tag_rows = result.filter(pl.col('variantId') == '1_200_C_G')
        assert tag_rows['isLead'].to_list() == [False]

    def test_row_count_equals_locus_size(self, credible_set: pl.LazyFrame) -> None:
        """Output must have one row per tag variant in the locus array."""
        result = process_credible_set(credible_set).collect()
        assert len(result) == 2

    def test_output_columns(self, credible_set: pl.LazyFrame) -> None:
        """Output must contain exactly the expected columns."""
        result = process_credible_set(credible_set).collect()
        expected = {
            'studyLocusId', 'studyId', 'pValueMantissa', 'pValueExponent',
            'beta', 'is95CredibleSet', 'is99CredibleSet', 'posteriorProbability',
            'variantId', 'finemappingMethod', 'isLead',
        }
        assert set(result.columns) == expected


class TestProcessL2G:
    """Tests for process_l2g."""

    @pytest.fixture
    def l2g(self) -> pl.LazyFrame:
        """Three genes for one locus: top-scorer, a strong secondary (>= 0.5), and a weak one."""
        return pl.DataFrame({
            'studyLocusId': ['locus1', 'locus1', 'locus1'],
            'geneId':       ['GENE_A', 'GENE_B', 'GENE_C'],
            'score':        [0.9, 0.6, 0.1],
        }).lazy()

    def test_top_gene_is_kept(self, l2g: pl.LazyFrame) -> None:
        """The highest-scoring gene per locus (rank == 1) must be present."""
        result = process_l2g(l2g).collect()
        assert 'GENE_A' in result['gwasGeneId'].to_list()

    def test_strong_secondary_gene_is_kept(self, l2g: pl.LazyFrame) -> None:
        """A non-top gene with score >= 0.5 must also be retained."""
        result = process_l2g(l2g).collect()
        assert 'GENE_B' in result['gwasGeneId'].to_list()

    def test_weak_non_top_gene_is_dropped(self, l2g: pl.LazyFrame) -> None:
        """A gene that is not rank-1 and has score < 0.5 must be excluded."""
        result = process_l2g(l2g).collect()
        assert 'GENE_C' not in result['gwasGeneId'].to_list()

    def test_output_columns_are_renamed(self, l2g: pl.LazyFrame) -> None:
        """Output must use gwasGeneId and gwasLocusToGeneScore column names."""
        result = process_l2g(l2g).collect()
        assert 'gwasGeneId' in result.columns
        assert 'gwasLocusToGeneScore' in result.columns
        assert 'geneId' not in result.columns
        assert 'score' not in result.columns
