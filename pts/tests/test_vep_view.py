"""Tests for the polars vep_view transformer."""

import gzip
import struct
from pathlib import Path

import polars as pl
import pysam
import pytest
from otter.config.model import Config

from pts.transformers.utils.dataset import write_dataset
from pts.transformers.vep_view import (
    COMMON_COLUMNS,
    GWAS_COLUMNS,
    QTL_COLUMNS,
    process_biosample,
    process_credible_set,
    process_l2g,
    process_study,
    vep_view,
)
from tests.credible_set_fixtures import published_credible_set, published_credible_set_variant


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
        """geneId and biosampleId must be renamed to qtlGeneId and qtlBiosampleId."""
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
    def credible_set(self) -> tuple[pl.LazyFrame, pl.LazyFrame]:
        """One credible set with two variants; the lead is the first one."""
        credible_set = published_credible_set({
            'id': 'cs1',
            'studyId': 'study1',
            'leadVariantId': '1_100_A_T',
            'fineMappingMethod': 'SuSiE',
        })
        credible_set_variant = published_credible_set_variant(
            {
                'credibleSetId': 'cs1',
                'variantId': '1_100_A_T',
                'pValueMantissa': 5.0,
                'pValueExponent': -8,
                'reportedBeta': 0.1,
                'conditionalBeta': 0.09,
                'posteriorProbability': 0.8,
            },
            {
                'credibleSetId': 'cs1',
                'variantId': '1_200_C_G',
                'pValueMantissa': 1.0,
                'pValueExponent': -5,
                'reportedBeta': 0.05,
                'posteriorProbability': 0.2,
            },
        )
        return credible_set.lazy(), credible_set_variant.lazy()

    def test_lead_variant_is_flagged(self, credible_set: tuple[pl.LazyFrame, pl.LazyFrame]) -> None:
        """The tag variant matching the locus lead variantId must have isLead=True."""
        result = process_credible_set(*credible_set).collect()
        assert isinstance(result, pl.DataFrame)
        lead_rows = result.filter(pl.col('variantId') == '1_100_A_T')
        assert lead_rows['isLead'].to_list() == [True]

    def test_tag_variant_not_flagged(self, credible_set: tuple[pl.LazyFrame, pl.LazyFrame]) -> None:
        """A tag variant that is not the lead must have isLead=False."""
        result = process_credible_set(*credible_set).collect()
        assert isinstance(result, pl.DataFrame)
        tag_rows = result.filter(pl.col('variantId') == '1_200_C_G')
        assert tag_rows['isLead'].to_list() == [False]

    def test_one_row_per_credible_set_variant(self, credible_set: tuple[pl.LazyFrame, pl.LazyFrame]) -> None:
        """Output must have one row per variant of the credible set."""
        result = process_credible_set(*credible_set).collect()
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 2

    def test_output_columns(self, credible_set: tuple[pl.LazyFrame, pl.LazyFrame]) -> None:
        """Output must contain exactly the expected columns."""
        result = process_credible_set(*credible_set).collect()
        assert isinstance(result, pl.DataFrame)
        expected = {
            'credibleSetId',
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

    def test_beta_is_the_reported_effect_and_every_variant_is_in_the_credible_intervals(
        self, credible_set: tuple[pl.LazyFrame, pl.LazyFrame]
    ) -> None:
        """The published variants are the 95% members, so both credible interval flags hold."""
        result = process_credible_set(*credible_set).collect().sort('variantId')
        assert result['beta'].to_list() == [0.1, 0.05]
        assert result['is95CredibleSet'].to_list() == [True, True]
        assert result['is99CredibleSet'].to_list() == [True, True]
        assert result['finemappingMethod'].to_list() == ['SuSiE', 'SuSiE']


class TestProcessL2G:
    """Tests for process_l2g."""

    @pytest.fixture
    def l2g(self) -> pl.LazyFrame:
        """Three genes for one locus: top-scorer, a strong secondary (>= 0.5), and a weak one."""
        return pl.DataFrame({
            'credibleSetId': ['cs1', 'cs1', 'cs1'],
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


def _variant(credible_set_id: str, variant_id: str, posterior: float) -> dict:
    return {
        'credibleSetId': credible_set_id,
        'variantId': variant_id,
        'pValueMantissa': 1.0,
        'pValueExponent': -10,
        'reportedBeta': 0.1,
        'posteriorProbability': posterior,
    }


@pytest.fixture
def outputs(tmp_path: Path) -> dict[str, str]:
    """Run vep_view on one GWAS locus (chr 2 and 10) and one QTL locus (chr 1)."""
    credible_set = published_credible_set(
        {'id': 'gwasLocus', 'studyId': 'gwasStudy', 'leadVariantId': '2_500_A_G', 'fineMappingMethod': 'SuSiE'},
        {'id': 'qtlLocus', 'studyId': 'qtlStudy', 'leadVariantId': '1_100_A_T', 'fineMappingMethod': 'SuSiE'},
    )
    credible_set_variant = published_credible_set_variant(
        _variant('gwasLocus', '2_500_A_G', 0.7),
        _variant('gwasLocus', '10_50_C_T', 0.3),
        _variant('qtlLocus', '1_100_A_T', 0.9),
        _variant('qtlLocus', '1_200_C_G', 0.1),
    )
    study = pl.DataFrame(
        {
            'studyId': ['gwasStudy', 'qtlStudy'],
            'studyType': ['gwas', 'eqtl'],
            'diseaseIds': [['EFO_1'], []],
            'geneId': [None, 'ENSG_QTL'],
            'biosampleId': [None, 'UBERON_1'],
        },
        schema_overrides={'diseaseIds': pl.List(pl.String)},
    )
    l2g = pl.DataFrame({'credibleSetId': ['gwasLocus'], 'geneId': ['ENSG_L2G'], 'score': [0.8]})
    biosample = pl.DataFrame({'biosampleId': ['UBERON_1'], 'biosampleName': ['liver']})

    source = {}
    for name, frame in {
        'credible_set': credible_set,
        'credible_set_variant': credible_set_variant,
        'study_table': study,
        'l2g_table': l2g,
        'biosample': biosample,
    }.items():
        source[name] = str(tmp_path / 'input' / name)
        write_dataset(frame, source[name])

    export = tmp_path / 'etc' / 'export' / 'vep_data'
    export.mkdir(parents=True)
    names = {'all': 'open_targets_vep', 'gwas': 'open_targets_vep_gwas', 'qtl': 'open_targets_vep_qtl'}
    destination = {f'vep_data_{key}': str(export / f'{name}.tsv.bgz') for key, name in names.items()}

    vep_view(source, destination, {}, Config(step='vep_view', steps=['vep_view'], work_path=tmp_path))
    return destination


class TestVepView:
    """End-to-end tests for vep_view: bgzipped, tabix-indexed output files."""

    @pytest.mark.parametrize(
        ('key', 'columns'),
        [
            ('vep_data_all', COMMON_COLUMNS + GWAS_COLUMNS + QTL_COLUMNS),
            ('vep_data_gwas', COMMON_COLUMNS + GWAS_COLUMNS),
            ('vep_data_qtl', COMMON_COLUMNS + QTL_COLUMNS),
        ],
    )
    def test_header_is_tabix_meta_line(self, outputs: dict[str, str], key: str, columns: list[str]) -> None:
        """The `#` header must be kept by the index (what `tabix -H` prints) with the file's columns."""
        assert Path(f'{outputs[key]}.tbi').exists()
        with pysam.TabixFile(outputs[key]) as tbx:
            assert list(tbx.header) == ['\t'.join(columns)]

    def test_all_file_indexes_every_chromosome(self, outputs: dict[str, str]) -> None:
        """The combined file must contain every chromosome and be fetchable by region."""
        with pysam.TabixFile(outputs['vep_data_all']) as tbx:
            assert sorted(tbx.contigs) == ['1', '10', '2']
            rows = [row.split('\t') for row in tbx.fetch('1', 150, 250)]
        assert [row[4] for row in rows] == ['1_200_C_G']

    def test_gwas_file_has_only_gwas_rows(self, outputs: dict[str, str]) -> None:
        """The GWAS file must hold the GWAS tags, with the L2G gene."""
        with pysam.TabixFile(outputs['vep_data_gwas']) as tbx:
            assert sorted(tbx.contigs) == ['10', '2']
            rows = [row.split('\t') for contig in tbx.contigs for row in tbx.fetch(contig)]
        gene = len(COMMON_COLUMNS) + GWAS_COLUMNS.index('gwasGeneId')
        assert {row[6] for row in rows} == {'gwas'}
        assert {row[gene] for row in rows} == {'ENSG_L2G'}

    def test_qtl_file_has_only_qtl_rows(self, outputs: dict[str, str]) -> None:
        """The QTL file must hold the QTL tags, with the measured gene and biosample name."""
        with pysam.TabixFile(outputs['vep_data_qtl']) as tbx:
            assert list(tbx.contigs) == ['1']
            rows = [row.split('\t') for row in tbx.fetch('1')]
        assert [row[4] for row in rows] == ['1_100_A_T', '1_200_C_G']
        assert {tuple(row[len(COMMON_COLUMNS) :]) for row in rows} == {('ENSG_QTL', 'liver')}

    def test_no_temporary_files_are_left(self, outputs: dict[str, str], tmp_path: Path) -> None:
        """The local TSV/bgzip scratch directory must be removed after upload."""
        assert not list(tmp_path.glob('vep_view_*'))


ENSEMBL_COMMON_HEADER = [
    '#chromosome',
    'position',
    'referenceAllele',
    'alternateAllele',
    'variantId',
    'studyId',
    'studyType',
    'credibleSetId',
    'pValueMantissa',
    'pValueExponent',
    'beta',
    'is95CredibleSet',
    'is99CredibleSet',
    'posteriorProbability',
    'finemappingMethod',
    'isLead',
]
ENSEMBL_GWAS_HEADER = ['gwasGeneId', 'gwasLocusToGeneScore', 'gwasDiseases']
ENSEMBL_QTL_HEADER = ['qtlGeneId', 'qtlBiosampleName']


def _rows(path: str) -> list[list[str]]:
    with pysam.TabixFile(path) as tbx:
        return [row.split('\t') for contig in tbx.contigs for row in tbx.fetch(contig)]


def _tabix_conf(path: str) -> tuple[int, int, int, int, str, int]:
    """Read (format, col_seq, col_beg, col_end, meta, skip) from a `.tbi` header, 1-based columns."""
    with gzip.open(f'{path}.tbi') as f:
        magic, _n_ref, fmt, col_seq, col_beg, col_end, meta, skip = struct.unpack('<4s7i', f.read(32))
    assert magic == b'TBI\x01'
    return fmt, col_seq, col_beg, col_end, chr(meta), skip


class TestEnsemblContract:
    """The file layout promised to the Ensembl VEP plugin (opentargets/issues#3462).

    The plugin reads column names from `tabix -H`. The split files are documented as
    `cut -f 1-16,17-19` (GWAS) and `cut -f 1-16,20,21` (QTL) of the combined file.
    The headers are literal lists on purpose: they must not follow the transformer's constants.
    """

    @pytest.mark.parametrize(
        ('key', 'header'),
        [
            ('vep_data_all', ENSEMBL_COMMON_HEADER + ENSEMBL_GWAS_HEADER + ENSEMBL_QTL_HEADER),
            ('vep_data_gwas', ENSEMBL_COMMON_HEADER + ENSEMBL_GWAS_HEADER),
            ('vep_data_qtl', ENSEMBL_COMMON_HEADER + ENSEMBL_QTL_HEADER),
        ],
    )
    def test_header(self, outputs: dict[str, str], key: str, header: list[str]) -> None:
        """Each file's `tabix -H` output is exactly the promised header."""
        with pysam.TabixFile(outputs[key]) as tbx:
            assert list(tbx.header) == ['\t'.join(header)]

    @pytest.mark.parametrize('key', ['vep_data_all', 'vep_data_gwas', 'vep_data_qtl'])
    def test_index_is_generic_tsv_on_chromosome_and_position(self, outputs: dict[str, str], key: str) -> None:
        """Each index is `tabix -s 1 -b 2 -e 2` with `#` meta lines and no skipped lines."""
        assert _tabix_conf(outputs[key]) == (0, 1, 2, 2, '#', 0)

    def test_gwas_file_is_cut_of_gwas_rows(self, outputs: dict[str, str]) -> None:
        """The GWAS file equals `cut -f 1-16,17-19` of the combined file's `gwas` rows."""
        combined = [row[:19] for row in _rows(outputs['vep_data_all']) if row[6] == 'gwas']
        assert combined
        assert _rows(outputs['vep_data_gwas']) == combined

    def test_qtl_file_is_cut_of_qtl_rows(self, outputs: dict[str, str]) -> None:
        """The QTL file equals `cut -f 1-16,20,21` of the combined file's rows whose study type ends in `qtl`."""
        combined = [row[:16] + row[19:21] for row in _rows(outputs['vep_data_all']) if row[6].endswith('qtl')]
        assert combined
        assert _rows(outputs['vep_data_qtl']) == combined
