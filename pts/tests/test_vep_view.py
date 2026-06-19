"""Tests for the polars vep_view transformer."""

import polars as pl
import pytest

from pts.transformers.vep_view import (
    process_biosample,
    process_credible_set,
    process_l2g,
    process_study,
)


class TestProcessBiosample:
    """Tests for process_biosample."""

    @pytest.fixture
    def biosample(self) -> pl.LazyFrame:
        """Two biosamples, one duplicated."""
        return pl.DataFrame({
            'biosampleId': ['BS1', 'BS2', 'BS1'],
            'biosampleName': ['Tissue A', 'Tissue B', 'Tissue A'],
        }).lazy()

    def test_columns_are_renamed(self, biosample: pl.LazyFrame) -> None:
        """Output must use qtlBiosampleId and qtlBiosampleName."""
        result = process_biosample(biosample).collect()
        assert isinstance(result, pl.DataFrame)
        assert 'qtlBiosampleId' in result.columns
        assert 'qtlBiosampleName' in result.columns
        assert 'biosampleId' not in result.columns
        assert 'biosampleName' not in result.columns

    def test_duplicates_are_removed(self, biosample: pl.LazyFrame) -> None:
        """Duplicate rows must be deduplicated."""
        result = process_biosample(biosample).collect()
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 2

    def test_values_are_preserved(self, biosample: pl.LazyFrame) -> None:
        """Biosample IDs and names must be unchanged after rename."""
        result = process_biosample(biosample).collect()
        assert isinstance(result, pl.DataFrame)
        result_sorted = result.sort('qtlBiosampleId')
        assert result_sorted['qtlBiosampleId'].to_list() == ['BS1', 'BS2']
        assert result_sorted['qtlBiosampleName'].to_list() == ['Tissue A', 'Tissue B']


class TestProcessStudy:
    """Tests for process_study."""

    @pytest.fixture
    def study(self) -> pl.LazyFrame:
        """Three studies: GWAS with diseases, GWAS with empty disease list, and a QTL."""
        return pl.DataFrame({
            'studyId': ['gwas1', 'gwas2', 'qtl1'],
            'studyType': ['gwas', 'gwas', 'eqtl'],
            'diseaseIds': [['EFO_001', 'EFO_002'], [], []],
            'geneId': [None, None, 'ENSG001'],
            'biosampleId': [None, None, 'BS1'],
        }).lazy()

    def test_gwas_diseases_joined_with_pipe(self, study: pl.LazyFrame) -> None:
        """Non-empty diseaseIds must be pipe-joined into gwasDiseases."""
        result = process_study(study).collect()
        assert isinstance(result, pl.DataFrame)
        row = result.filter(pl.col('studyId') == 'gwas1')
        assert row['gwasDiseases'].to_list() == ['EFO_001|EFO_002']

    def test_empty_disease_list_is_null(self, study: pl.LazyFrame) -> None:
        """An empty diseaseIds list must produce a null gwasDiseases value."""
        result = process_study(study).collect()
        assert isinstance(result, pl.DataFrame)
        row = result.filter(pl.col('studyId') == 'gwas2')
        assert row['gwasDiseases'].to_list() == [None]

    def test_qtl_columns_are_renamed(self, study: pl.LazyFrame) -> None:
        """geneId and biosampleId must be renamed to qtlGeneId and qtlBiosampleId."""  # noqa: D403
        result = process_study(study).collect()
        assert isinstance(result, pl.DataFrame)
        assert 'qtlGeneId' in result.columns
        assert 'qtlBiosampleId' in result.columns
        assert 'geneId' not in result.columns
        assert 'biosampleId' not in result.columns

    def test_qtl_gene_and_biosample_values(self, study: pl.LazyFrame) -> None:
        """QTL study must carry through its geneId and biosampleId values."""
        result = process_study(study).collect()
        assert isinstance(result, pl.DataFrame)
        row = result.filter(pl.col('studyId') == 'qtl1')
        assert row['qtlGeneId'].to_list() == ['ENSG001']
        assert row['qtlBiosampleId'].to_list() == ['BS1']


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
        assert isinstance(result, pl.DataFrame)
        lead_rows = result.filter(pl.col('variantId') == '1_100_A_T')
        assert lead_rows['isLead'].to_list() == [True]

    def test_tag_variant_not_flagged(self, credible_set: pl.LazyFrame) -> None:
        """A tag variant that is not the lead must have isLead=False."""
        result = process_credible_set(credible_set).collect()
        assert isinstance(result, pl.DataFrame)
        tag_rows = result.filter(pl.col('variantId') == '1_200_C_G')
        assert tag_rows['isLead'].to_list() == [False]

    def test_row_count_equals_locus_size(self, credible_set: pl.LazyFrame) -> None:
        """Output must have one row per tag variant in the locus array."""
        result = process_credible_set(credible_set).collect()
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 2

    def test_output_columns(self, credible_set: pl.LazyFrame) -> None:
        """Output must contain exactly the expected columns."""
        result = process_credible_set(credible_set).collect()
        assert isinstance(result, pl.DataFrame)
        expected = {
            'studyLocusId',
            'studyId',
            'pValueMantissa',
            'pValueExponent',
            'beta',
            'is95CredibleSet',
            'is99CredibleSet',
            'posteriorProbability',
            'variantId',
            'finemappingMethod',
            'isLead',
        }
        assert set(result.columns) == expected


class TestProcessL2G:
    """Tests for process_l2g."""

    @pytest.fixture
    def l2g(self) -> pl.LazyFrame:
        """Three genes for one locus: top-scorer, a strong secondary (>= 0.5), and a weak one."""
        return pl.DataFrame({
            'studyLocusId': ['locus1', 'locus1', 'locus1'],
            'geneId': ['GENE_A', 'GENE_B', 'GENE_C'],
            'score': [0.9, 0.6, 0.1],
        }).lazy()

    def test_top_gene_is_kept(self, l2g: pl.LazyFrame) -> None:
        """The highest-scoring gene per locus (rank == 1) must be present."""
        result = process_l2g(l2g).collect()
        assert isinstance(result, pl.DataFrame)
        assert 'GENE_A' in result['gwasGeneId'].to_list()

    def test_strong_secondary_gene_is_kept(self, l2g: pl.LazyFrame) -> None:
        """A non-top gene with score >= 0.5 must also be retained."""
        result = process_l2g(l2g).collect()
        assert isinstance(result, pl.DataFrame)
        assert 'GENE_B' in result['gwasGeneId'].to_list()

    def test_weak_non_top_gene_is_dropped(self, l2g: pl.LazyFrame) -> None:
        """A gene that is not rank-1 and has score < 0.5 must be excluded."""
        result = process_l2g(l2g).collect()
        assert isinstance(result, pl.DataFrame)
        assert 'GENE_C' not in result['gwasGeneId'].to_list()

    def test_output_columns_are_renamed(self, l2g: pl.LazyFrame) -> None:
        """Output must use gwasGeneId and gwasLocusToGeneScore column names."""
        result = process_l2g(l2g).collect()
        assert isinstance(result, pl.DataFrame)
        assert 'gwasGeneId' in result.columns
        assert 'gwasLocusToGeneScore' in result.columns
        assert 'geneId' not in result.columns
        assert 'score' not in result.columns
