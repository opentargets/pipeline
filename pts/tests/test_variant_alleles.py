"""VEP allele form and genomic HGVS tests."""

import gzip
from functools import cache
from pathlib import Path

import polars as pl
import pytest

from pts.transformers.variant.alleles import (
    contig_spans,
    hgvs_genomic,
    read_chromosome,
    read_contig,
    vep_allele_form,
    write_fai,
)

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


def test_contigs_of_the_uncompressed_fasta_match_read_chromosome(tmp_path: Path):
    plain = tmp_path / 'reference.fa'
    plain.write_bytes(gzip.decompress(Path(FASTA).read_bytes()))
    spans = contig_spans(plain)
    assert sorted(spans) == sorted(WINDOW_START)
    for chromosome, span in spans.items():
        assert read_contig(plain, span) == read_chromosome(FASTA, chromosome)


def test_contig_spans_keep_the_first_contig_of_a_name_and_a_last_line_without_newline(tmp_path: Path):
    plain = tmp_path / 'reference.fa'
    plain.write_bytes(b'>1 dna:chromosome\nacgt\nAC\n>2\nGG\n>1\nTTTT\n>3\nCA')
    spans = contig_spans(plain)
    assert {name: read_contig(plain, span) for name, span in spans.items()} == {'1': b'ACGTAC', '2': b'GG', '3': b'CA'}


@pytest.mark.parametrize(
    ('fasta', 'expected'),
    [
        pytest.param(
            b'>1 dna:chromosome\nACGT\nACGT\nAC\n>2\nGGGG\n>3\nTTTT\nTT\n',
            ['1\t10\t18\t4\t5', '2\t4\t34\t4\t5', '3\t6\t42\t4\t5'],
            id='short-last-line-and-one-full-line',
        ),
        pytest.param(b'>1\nACGT\nAC', ['1\t6\t3\t4\t5'], id='last-line-without-newline'),
        pytest.param(b'>1\r\nACGT\r\nAC\r\n>2\r\nGG\r\n', ['1\t6\t4\t4\t6', '2\t2\t18\t2\t4'], id='crlf'),
        pytest.param(b'>1\n>2\nAC\n', ['1\t0\t3\t0\t0', '2\t2\t6\t2\t3'], id='empty-contig'),
        pytest.param(b'>1\nAC\n>2\nGG\n>1\nTTTT\n', ['1\t2\t3\t2\t3', '2\t2\t9\t2\t3'], id='first-of-a-name'),
    ],
)
def test_write_fai(tmp_path: Path, fasta: bytes, expected: list[str]):
    plain = tmp_path / 'reference.fa'
    plain.write_bytes(fasta)
    fai = write_fai(plain, contig_spans(plain))
    assert fai == tmp_path / 'reference.fa.fai'
    assert fai.read_text().splitlines() == expected
    assert sorted(p.name for p in tmp_path.iterdir()) == ['reference.fa', 'reference.fa.fai']


@pytest.mark.parametrize(
    'fasta',
    [
        pytest.param(b'>1\nACGT\nAC\nACGT\n>2\nGG\n', id='short-line-inside'),
        pytest.param(b'>1\nAC\nACGT\n', id='long-last-line'),
        pytest.param(b'>1\nACGT\n\nACGT\n', id='blank-line'),
    ],
)
def test_write_fai_refuses_uneven_lines(tmp_path: Path, fasta: bytes):
    plain = tmp_path / 'reference.fa'
    plain.write_bytes(fasta)
    (tmp_path / 'reference.fa.fai').write_text('stale\n')
    with pytest.raises(ValueError, match=r'contig 1 .*uneven'):
        write_fai(plain, contig_spans(plain))
    assert (tmp_path / 'reference.fa.fai').read_text() == 'stale\n'
    assert sorted(p.name for p in tmp_path.iterdir()) == ['reference.fa', 'reference.fa.fai']


def test_fai_rows_address_the_contigs_read_contig_reads(tmp_path: Path):
    plain = tmp_path / 'reference.fa'
    plain.write_bytes(gzip.decompress(Path(FASTA).read_bytes()))
    spans = contig_spans(plain)
    data = plain.read_bytes()
    for line in write_fai(plain, spans).read_text().splitlines():
        name, length, offset, bases, width = line.split('\t')
        length, offset, bases, width = int(length), int(offset), int(bases), int(width)
        lines = [data[offset + i * width : offset + i * width + bases] for i in range(-(-length // bases))]
        assert b''.join(lines)[:length].upper() == read_contig(plain, spans[name])
