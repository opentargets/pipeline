"""Tests for the transcript fields the variant step takes from the Ensembl core tables."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from pts.transformers.variant.transcripts import transcript_fields
from tests.ensembl_fixtures import cells, ddl, write_dump

#: Table name to (columns in DDL order, rows). Transcripts 301752 and 304535 are the human miRNAs
#: ENST00000585066 and ENST00000583386 with their release 116 `miRNA` attribute values and gene
#: coordinates; the others are synthetic.
TABLES: dict[str, tuple[list[str], list[tuple[object, ...]]]] = {
    'coord_system': (['coord_system_id', 'species_id', 'name'], [(1, 1, 'chromosome')]),
    'seq_region': (
        ['seq_region_id', 'name', 'coord_system_id', 'length'],
        [(131557, '22', 1, 50818468), (10, '1', 1, 1000)],
    ),
    'gene': (
        [
            'gene_id',
            'biotype',
            'seq_region_id',
            'seq_region_start',
            'seq_region_end',
            'seq_region_strand',
            'display_xref_id',
            'description',
            'canonical_transcript_id',
            'stable_id',
        ],
        [
            (1, 'protein_coding', 131557, 1000, 5000, 1, 501, None, 100, 'ENSG01'),
            (2, 'miRNA', 131557, 19963753, 19963834, 1, 502, None, 301752, 'ENSG02'),
            (3, 'miRNA', 131557, 31160062, 31160119, -1, 503, None, 304535, 'ENSG03'),
            (4, 'lncRNA', 131557, 7000, 7400, -1, 504, None, 400, 'ENSG04'),
            (5, 'protein_coding', 10, 100, 900, 1, 505, None, 500, 'ENSG05'),
            (6, 'lncRNA', 131557, 9000, 9100, 1, None, None, 600, 'ENSG06'),
        ],
    ),
    'transcript': (
        [
            'transcript_id',
            'gene_id',
            'seq_region_id',
            'seq_region_start',
            'seq_region_end',
            'seq_region_strand',
            'biotype',
            'canonical_translation_id',
            'stable_id',
        ],
        [
            (100, 1, 131557, 1000, 5000, 1, 'protein_coding', None, 'ENST_CODING'),
            (101, 1, 131557, 1200, 4000, 1, 'protein_coding', None, 'ENST_PLAIN'),
            (301752, 2, 131557, 19963753, 19963834, 1, 'miRNA', None, 'ENST00000585066'),
            (304535, 3, 131557, 31160062, 31160119, -1, 'miRNA', None, 'ENST00000583386'),
            (400, 4, 131557, 7000, 7400, -1, 'lncRNA', None, 'ENST_SPLICED_MINUS'),
            (401, 4, 131557, 7000, 7400, -1, 'lncRNA', None, 'ENST_READTHROUGH'),
            (500, 5, 10, 100, 900, 1, 'protein_coding', None, 'ENST_CHR1'),
            (600, 6, 131557, 9000, 9100, 1, 'lncRNA', None, 'ENST_NO_SYMBOL'),
        ],
    ),
    'translation': (
        ['translation_id', 'transcript_id', 'stable_id'],
        [(1000, 100, 'ENSP_CODING')],
    ),
    'exon': (
        ['exon_id', 'seq_region_id', 'seq_region_start', 'seq_region_end', 'seq_region_strand', 'stable_id'],
        [
            (1, 131557, 1000, 5000, 1, 'ENSE1'),
            (2, 131557, 19963753, 19963834, 1, 'ENSE2'),
            (3, 131557, 31160062, 31160119, -1, 'ENSE3'),
            (4, 131557, 7300, 7400, -1, 'ENSE4'),
            (5, 131557, 7000, 7050, -1, 'ENSE5'),
        ],
    ),
    'exon_transcript': (
        ['exon_id', 'transcript_id', 'rank'],
        [(1, 100, 1), (2, 301752, 1), (3, 304535, 1), (5, 400, 2), (4, 400, 1)],
    ),
    'attrib_type': (
        ['attrib_type_id', 'code', 'name'],
        [
            (15, 'miRNA', 'Micro RNA'),
            (218, 'readthrough_tra', 'readthrough transcript'),
            (427, 'appris', 'APPRIS'),
            (535, 'MANE_Select', 'MANE Select'),
            (554, 'is_canonical', 'Ensembl Canonical'),
            (576, 'gencode_primary', 'GENCODE primary annotation'),
        ],
    ),
    'transcript_attrib': (
        ['transcript_id', 'attrib_type_id', 'value'],
        [
            (100, 427, 'principal1'),
            (100, 535, 'NM_000001.1'),
            (100, 554, '1'),
            (101, 427, 'alternative2'),
            (301752, 15, '3-23'),
            (301752, 15, '62-82'),
            (304535, 15, '3-25'),
            (304535, 15, '37-58'),
            # cDNA 90-120 of the minus strand transcript spans both exons (101 bases, then 51).
            (400, 15, '90-120'),
            (401, 218, 'readthrough'),
            (401, 15, '1-10'),
        ],
    ),
    'external_db': (
        ['external_db_id', 'db_name'],
        [(2200, 'Uniprot/SWISSPROT'), (2000, 'Uniprot/SPTREMBL'), (1100, 'HGNC')],
    ),
    'xref': (
        ['xref_id', 'external_db_id', 'dbprimary_acc', 'display_label'],
        [
            (501, 1100, 'HGNC:1', 'GENEA'),
            (502, 1100, 'HGNC:2', 'MIR646'),
            (503, 1100, 'HGNC:3', 'MIR1281'),
            (504, 1100, 'HGNC:4', 'LNCX'),
            (505, 1100, 'HGNC:5', 'GENEONE'),
            (601, 2200, 'P00001', 'P00001'),
            (602, 2000, 'A0A001', 'A0A001'),
        ],
    ),
    'object_xref': (
        ['object_xref_id', 'ensembl_id', 'ensembl_object_type', 'xref_id'],
        [(1, 1000, 'Translation', 601), (2, 1000, 'Translation', 602)],
    ),
    'analysis': (['analysis_id', 'created', 'logic_name'], [(30, None, 'signalp')]),
    'protein_feature': (['protein_feature_id', 'translation_id', 'hit_name', 'analysis_id'], [(1, 1000, 'x', 30)]),
}


@pytest.fixture
def core(tmp_path: Path) -> str:
    schema_ddl = ddl({name: columns for name, (columns, _) in TABLES.items()})
    rows = {name: [cells(*row) for row in table_rows] for name, (_, table_rows) in TABLES.items()}
    return write_dump(tmp_path, rows, schema_ddl)


def _row(core: str, transcript_id: str, chromosome: str = '22') -> dict:
    frame = transcript_fields(core, chromosome)
    return frame.filter(pl.col('transcript_id') == transcript_id).row(0, named=True)


def test_appris_mane_canonical(core: str) -> None:
    row = _row(core, 'ENST_CODING')
    assert row['appris'] == 'principal1'
    assert row['mane_select'] == 'NM_000001.1'
    assert row['canonical'] == 1
    assert row['gene_symbol'] == 'GENEA'
    assert row['biotype'] == 'protein_coding'
    assert row['swissprot'] == ['P00001']
    assert row['trembl'] == ['A0A001']


def test_transcript_without_flags(core: str) -> None:
    row = _row(core, 'ENST_PLAIN')
    assert row['appris'] == 'alternative2'
    assert row['mane_select'] is None
    assert row['canonical'] is None
    assert row['swissprot'] == []
    assert row['trembl'] == []
    assert row['mature_mirna'] == []


def test_gene_without_symbol(core: str) -> None:
    assert _row(core, 'ENST_NO_SYMBOL')['gene_symbol'] is None


def test_geometry(core: str) -> None:
    plus = _row(core, 'ENST_CODING')
    assert (plus['transcript_start'], plus['transcript_end'], plus['strand'], plus['tss']) == (1000, 5000, 1, 1000)
    minus = _row(core, 'ENST_SPLICED_MINUS')
    assert (minus['transcript_start'], minus['transcript_end'], minus['strand'], minus['tss']) == (7000, 7400, -1, 7400)


def test_schema(core: str) -> None:
    assert transcript_fields(core, '22').schema == pl.Schema({
        'transcript_id': pl.String(),
        'canonical': pl.Int64(),
        'appris': pl.String(),
        'mane_select': pl.String(),
        'gene_symbol': pl.String(),
        'biotype': pl.String(),
        'swissprot': pl.List(pl.String()),
        'trembl': pl.List(pl.String()),
        'tss': pl.Int64(),
        'transcript_start': pl.Int64(),
        'transcript_end': pl.Int64(),
        'strand': pl.Int8(),
        'mature_mirna': pl.List(pl.Struct({'start': pl.Int64(), 'end': pl.Int64()})),
    })


def test_mature_mirna_coordinates_plus_strand(core: str) -> None:
    mature = _row(core, 'ENST00000585066')['mature_mirna']
    assert mature == [{'start': 19963755, 'end': 19963775}, {'start': 19963814, 'end': 19963834}]
    assert any(m['start'] <= 19963756 <= m['end'] for m in mature)


def test_mature_mirna_coordinates_minus_strand(core: str) -> None:
    mature = _row(core, 'ENST00000583386')['mature_mirna']
    assert mature == [{'start': 31160062, 'end': 31160083}, {'start': 31160095, 'end': 31160117}]
    assert [any(m['start'] <= position <= m['end'] for m in mature) for position in (31160099, 31160117)] == [True] * 2


def test_mature_mirna_spanning_exons_is_split_at_the_intron(core: str) -> None:
    mature = _row(core, 'ENST_SPLICED_MINUS')['mature_mirna']
    assert mature == [{'start': 7032, 'end': 7050}, {'start': 7300, 'end': 7311}]


def test_readthrough_transcripts_absent(core: str) -> None:
    assert 'ENST_READTHROUGH' not in transcript_fields(core, '22')['transcript_id'].to_list()


def test_only_the_requested_chromosome(core: str) -> None:
    assert transcript_fields(core, '1')['transcript_id'].to_list() == ['ENST_CHR1']
