"""VEP allele form and genomic HGVS tests."""

from functools import cache
from pathlib import Path

import polars as pl
import pytest

from pts.transformers.variant.alleles import hgvs_genomic, read_chromosome, vep_allele_form

#: Ensembl 116 GRCh38 bases around each test variant, one contig per chromosome. Each contig
#: holds only a window, whose first position is in its header and in `WINDOW_START`.
FASTA = str(Path(__file__).parent / 'test_data' / 'variant_reference.fa.gz')
WINDOW_START = {
    '4': 27501,
    '11': 50098131,
    '13': 16407251,
    '16': 86801,
    '18': 16226031,
    '22': 16848801,
    'MT': 1,
    'X': 1464201,
}


@cache
def reference(chromosome: str) -> bytes:
    """The chromosome window, left-padded with N so that byte i is position i + 1."""
    return b'N' * (WINDOW_START[chromosome] - 1) + read_chromosome(FASTA, chromosome)


def _allele_form(variant_ids: list[str]) -> list[tuple]:
    parts = [variant_id.split('_') for variant_id in variant_ids]
    frame = pl.LazyFrame(
        {
            'chromosome': [p[0] for p in parts],
            'position': [int(p[1]) for p in parts],
            'referenceAllele': [p[2] for p in parts],
            'alternateAllele': [p[3] for p in parts],
        },
        schema_overrides={'position': pl.Int32},
    )
    result = vep_allele_form(frame).collect()
    assert result.schema['vepStart'] == pl.Int64
    assert result.schema['vepEnd'] == pl.Int64
    assert result.schema['vepReference'] == pl.String
    assert result.schema['vepAlternate'] == pl.String
    return result.select('vepStart', 'vepEnd', 'vepReference', 'vepAlternate').rows()


def test_read_chromosome_returns_one_contig_in_upper_case():
    assert read_chromosome(FASTA, 'MT')[:15] == b'GATCACAGGTCTATC'
    assert len(read_chromosome(FASTA, 'MT')) == 600
    assert len(read_chromosome(FASTA, '22')) == 1100
    assert read_chromosome(FASTA, '18') == read_chromosome(FASTA, '18').upper()
    assert read_chromosome(FASTA, '18')[:10] == b'GAAACACTCT'


def test_read_chromosome_matches_the_contig_name_exactly():
    with pytest.raises(ValueError, match='1'):
        read_chromosome(FASTA, '1')


def test_vep_allele_form():
    assert _allele_form([
        '1_7741944_GTT_GT',
        '22_17174835_CAAA_CA',
        '1_170333597_T_TAT',
        '18_16226056_A_T',
    ]) == [
        (7741946, 7741946, 'T', '-'),
        (17174837, 17174838, 'AA', '-'),
        (170333598, 170333597, '-', 'AT'),
        (16226056, 16226056, 'A', 'T'),
    ]


def test_equal_length_pairs_stay_untrimmed():
    assert _allele_form(['10_94936017_CTG_CGT']) == [(94936017, 94936019, 'CTG', 'CGT')]


def test_vep_allele_form_keeps_rows_and_columns():
    frame = pl.LazyFrame({
        'chromosome': ['2', '1'],
        'position': [5, 3],
        'referenceAllele': ['ACGT', 'A'],
        'alternateAllele': ['AT', 'ACGTACGT'],
        'rsId': ['rs2', '.'],
    })
    result = vep_allele_form(frame).collect()
    assert result.columns == [*frame.collect_schema().names(), 'vepStart', 'vepEnd', 'vepReference', 'vepAlternate']
    assert result.select('rsId', 'vepStart', 'vepEnd', 'vepReference', 'vepAlternate').rows() == [
        ('rs2', 6, 7, 'CG', '-'),
        ('.', 4, 3, '-', 'CGTACGT'),
    ]


# Expected values are the baseline's `hgvsId` (`do/platform-2610-2/output/variant`), except
# 11_50098155_A_T, which only the PPP run `sz/ppp-2610-1` has.
@pytest.mark.parametrize(
    ('variant_id', 'expected'),
    [
        pytest.param('MT_15_C_T', 'MT:m.15C>T', id='mt-substitution'),
        pytest.param('MT_513_GCA_G', 'MT:m.523_524del', id='mt-deletion-shifted'),
        pytest.param('11_50098155_A_T', '11:g.50098155C>T', id='ref-mismatch-uses-reference-base'),
        pytest.param('X_1464227_A_G', None, id='ref-mismatch-reference-equals-alt'),
        pytest.param('22_16849850_CA_C', '22:g.16849867del', id='deletion-shifted'),
        pytest.param('22_16848841_C_CCT', '22:g.16848863_16848864dup', id='insertion-in-repeat-dup'),
        pytest.param('4_27540_G_GGC', '4:g.27541_27542insCG', id='insertion-shifted-rotated'),
        pytest.param('16_86805_CG_GA', '16:g.86805_86806delinsGA', id='delins'),
        pytest.param('16_86889_TG_CA', '16:g.86889_86890inv', id='inversion'),
        pytest.param('18_16226056_A_T', '18:g.16226056A>T', id='intergenic-substitution'),
        pytest.param('13_16407276_GA_G', '13:g.16407277del', id='intergenic-deletion'),
    ],
)
def test_hgvs_genomic(variant_id, expected):
    chromosome, position, ref, alt = variant_id.split('_')
    assert hgvs_genomic(reference(chromosome), chromosome, int(position), ref, alt) == expected
