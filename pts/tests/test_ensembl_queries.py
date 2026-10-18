"""Tests for the shared Ensembl core dump queries."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from pts.schemas import ensembl_core as schema
from pts.transformers.utils.ensembl import CoreDump, gene_names, genes, transcripts, translation_xrefs
from pts.transformers.utils.ensembl.queries import CANONICAL_CHROMOSOMES
from tests.ensembl_fixtures import cells, ddl, write_dump

#: Table name to (columns in DDL order, rows). `None` is written as the dump's `\N`. Each table
#: carries one column the queries never read, so a positional misread shows up as a wrong value.
TABLES: dict[str, tuple[list[str], list[tuple[object, ...]]]] = {
    'coord_system': (
        ['coord_system_id', 'species_id', 'name'],
        [(1, 1, 'chromosome'), (2, 1, 'scaffold')],
    ),
    'seq_region': (
        ['seq_region_id', 'name', 'coord_system_id', 'length'],
        [(10, '1', 1, 1000), (11, 'KI270728.1', 2, 500), (12, 'MT', 1, 100), (13, 'X', 1, 1000)],
    ),
    'gene': (
        [
            'gene_id',
            'biotype',
            'analysis_id',
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
            (1, 'protein_coding', 9, 10, 100, 900, 1, 501, 'gene one', 102, 'ENSG01'),
            (2, 'lncRNA', 9, 13, 300, 400, -1, None, None, 201, 'ENSG02'),
            (3, 'protein_coding', 9, 11, 10, 20, 1, 503, 'on a scaffold', 301, 'ENSG03'),
            (4, 'protein_coding', 9, 12, 5, 60, 1, 504, 'mito', 401, 'ENSG04'),
            (5, 'protein_coding', 9, 10, 700, 800, 1, 505, '', 999, 'ENSG05'),
        ],
    ),
    'transcript': (
        [
            'transcript_id',
            'gene_id',
            'analysis_id',
            'seq_region_id',
            'seq_region_start',
            'seq_region_end',
            'seq_region_strand',
            'biotype',
            'canonical_translation_id',
            'stable_id',
        ],
        [
            (102, 1, 9, 10, 150, 900, 1, 'protein_coding', 1002, 'ENST02'),
            (101, 1, 9, 10, 100, 800, 1, 'retained_intron', None, 'ENST01'),
            (201, 2, 9, 13, 300, 400, -1, 'lncRNA', None, 'ENST03'),
            (301, 3, 9, 11, 10, 20, 1, 'protein_coding', 3001, 'ENST04'),
            (401, 4, 9, 12, 5, 60, 1, 'protein_coding', 4001, 'ENST05'),
        ],
    ),
    'translation': (
        ['translation_id', 'transcript_id', 'seq_start', 'stable_id'],
        [(1002, 102, 1, 'ENSP02'), (3001, 301, 1, 'ENSP04'), (4001, 401, 1, 'ENSP05')],
    ),
    'exon': (
        ['exon_id', 'seq_region_id', 'seq_region_start', 'seq_region_end', 'seq_region_strand', 'phase', 'stable_id'],
        [
            (7001, 10, 700, 900, 1, 0, 'ENSE01'),
            (7002, 10, 150, 300, 1, 0, 'ENSE02'),
            (7003, 13, 350, 400, -1, -1, 'ENSE03'),
            (7004, 13, 300, 320, -1, -1, 'ENSE04'),
        ],
    ),
    'exon_transcript': (
        ['exon_id', 'transcript_id', 'rank'],
        [(7001, 102, 2), (7002, 102, 1), (7003, 201, 1), (7004, 201, 2)],
    ),
    'attrib_type': (
        ['attrib_type_id', 'code', 'name'],
        [
            (1, 'is_canonical', 'canonical'),
            (2, 'MANE_Select', 'MANE Select'),
            (3, 'appris', 'APPRIS'),
            (4, 'gencode_primary', 'GENCODE primary'),
            (5, 'MANE_Plus_Clinical', 'MANE Plus Clinical'),
            (6, 'TSL', 'transcript support level'),
        ],
    ),
    'transcript_attrib': (
        ['transcript_id', 'attrib_type_id', 'value'],
        [
            (102, 3, 'principal1'),
            (102, 6, 'tsl1'),
            (102, 2, 'NM_000001.1'),
            (102, 1, '1'),
            (102, 4, '1'),
            (201, 5, 'NM_000002.1'),
        ],
    ),
    'external_db': (
        ['external_db_id', 'db_name', 'status'],
        [
            (2200, 'Uniprot/SWISSPROT', 'KNOWN'),
            (2000, 'Uniprot/SPTREMBL', 'XREF'),
            (2201, 'Uniprot_isoform', 'XREF'),
            (1100, 'HGNC', 'KNOWNXREF'),
        ],
    ),
    'xref': (
        ['xref_id', 'external_db_id', 'dbprimary_acc', 'display_label', 'version'],
        [
            (501, 1100, 'HGNC:1', 'GENEA', '0'),
            (503, 1100, 'HGNC:3', 'SCAFF', '0'),
            (504, 1100, 'HGNC:4', 'MT-CO1', '0'),
            (505, 1100, 'HGNC:5', 'GENEE', '0'),
            (601, 2200, 'P00002', 'P00002', '1'),
            (602, 2200, 'P00001', 'P00001', '1'),
            (603, 2000, 'A0A001', 'A0A001', '1'),
            (604, 2201, 'P00001-2', 'P00001-2', '1'),
            (605, 2200, 'Q00009', 'Q00009', '1'),
        ],
    ),
    'object_xref': (
        ['object_xref_id', 'ensembl_id', 'ensembl_object_type', 'xref_id'],
        [
            (1, 1002, 'Translation', 601),
            (2, 1002, 'Translation', 602),
            (3, 1002, 'Translation', 602),
            (4, 1002, 'Translation', 603),
            (5, 1002, 'Translation', 604),
            (6, 1002, 'Transcript', 605),
            (7, 1, 'Gene', 501),
        ],
    ),
    'analysis': (
        ['analysis_id', 'created', 'logic_name'],
        [(30, None, 'signalp'), (31, None, 'alphafold'), (32, None, 'pfam')],
    ),
    'protein_feature': (
        ['protein_feature_id', 'translation_id', 'hit_name', 'analysis_id'],
        [
            (1, 1002, 'SignalP-noTM', 30),
            (2, 1002, 'AF-P00001-F1', 31),
            (3, 1002, 'PF00001', 32),
            (4, 1002, 'SignalP-noTM', 30),
        ],
    ),
}


@pytest.fixture
def dump(tmp_path: Path) -> CoreDump:
    schema_ddl = ddl({name: columns for name, (columns, _) in TABLES.items()})
    rows = {name: [cells(*row) for row in table_rows] for name, (_, table_rows) in TABLES.items()}
    return CoreDump(write_dump(tmp_path, rows, schema_ddl))


def test_canonical_chromosomes() -> None:
    assert [*(str(i) for i in range(1, 23)), 'X', 'Y', 'MT'] == CANONICAL_CHROMOSOMES


def test_gene_names_covers_every_gene_including_scaffolds(dump: CoreDump) -> None:
    frame = gene_names(dump)
    assert frame.schema == schema.gene_names
    assert frame.rows() == [
        ('ENSG01', 'GENEA'),
        ('ENSG02', None),
        ('ENSG03', 'SCAFF'),
        ('ENSG04', 'MT-CO1'),
        ('ENSG05', 'GENEE'),
    ]


def test_genes_keeps_canonical_chromosomes_only(dump: CoreDump) -> None:
    frame = genes(dump)
    assert frame.schema == schema.genes
    assert frame['id'].to_list() == ['ENSG01', 'ENSG02', 'ENSG04', 'ENSG05']


def test_genes_values(dump: CoreDump) -> None:
    rows = {row['id']: row for row in genes(dump).iter_rows(named=True)}
    assert rows['ENSG01'] == {
        'id': 'ENSG01',
        'biotype': 'protein_coding',
        'description': 'gene one',
        'chromosome': '1',
        'start': 100,
        'end': 900,
        'strand': 1,
        'approvedSymbol': 'GENEA',
        'canonicalTranscriptId': 'ENST02',
    }
    assert rows['ENSG02']['strand'] == -1
    assert rows['ENSG02']['chromosome'] == 'X'
    assert rows['ENSG02']['approvedSymbol'] is None
    assert rows['ENSG02']['description'] is None
    assert rows['ENSG04']['chromosome'] == 'MT'
    assert rows['ENSG05']['description'] == ''


def test_gene_with_a_dangling_canonical_transcript_is_kept(dump: CoreDump) -> None:
    row = genes(dump).filter(pl.col('id') == 'ENSG05').row(0, named=True)
    assert row['canonicalTranscriptId'] is None


def test_gene_without_transcripts_has_no_transcript_rows(dump: CoreDump) -> None:
    assert transcripts(dump).filter(pl.col('geneId') == 'ENSG05').is_empty()


def test_transcripts_of_canonical_genes_sorted(dump: CoreDump) -> None:
    frame = transcripts(dump)
    assert frame.schema == schema.transcripts
    assert frame.select('geneId', 'transcriptId').rows() == [
        ('ENSG01', 'ENST01'),
        ('ENSG01', 'ENST02'),
        ('ENSG02', 'ENST03'),
        ('ENSG04', 'ENST05'),
    ]


def test_transcript_values(dump: CoreDump) -> None:
    row = transcripts(dump).filter(pl.col('transcriptId') == 'ENST02').row(0, named=True)
    assert row == {
        'geneId': 'ENSG01',
        'transcriptId': 'ENST02',
        'biotype': 'protein_coding',
        'chromosome': '1',
        'start': 150,
        'end': 900,
        'strand': 1,
        'translationId': 'ENSP02',
        'attributes': [
            {'code': 'MANE_Select', 'value': 'NM_000001.1'},
            {'code': 'appris', 'value': 'principal1'},
            {'code': 'gencode_primary', 'value': '1'},
            {'code': 'is_canonical', 'value': '1'},
        ],
        'exons': [
            {'exonId': 'ENSE02', 'chromosome': '1', 'start': 150, 'end': 300, 'strand': 1},
            {'exonId': 'ENSE01', 'chromosome': '1', 'start': 700, 'end': 900, 'strand': 1},
        ],
    }


def test_transcript_without_translation_attributes_or_exons(dump: CoreDump) -> None:
    row = transcripts(dump).filter(pl.col('transcriptId') == 'ENST01').row(0, named=True)
    assert row['translationId'] is None
    assert row['attributes'] == []
    assert row['exons'] == []


def test_negative_strand_exons_follow_rank(dump: CoreDump) -> None:
    row = transcripts(dump).filter(pl.col('transcriptId') == 'ENST03').row(0, named=True)
    assert row['strand'] == -1
    assert [exon['exonId'] for exon in row['exons']] == ['ENSE03', 'ENSE04']
    assert {exon['strand'] for exon in row['exons']} == {-1}
    assert row['attributes'] == [{'code': 'MANE_Plus_Clinical', 'value': 'NM_000002.1'}]


def test_translation_xrefs_one_row_per_translation(dump: CoreDump) -> None:
    frame = translation_xrefs(dump)
    assert frame.schema == schema.translation_xrefs
    assert frame.select('transcriptId', 'translationId').rows() == [('ENST02', 'ENSP02'), ('ENST05', 'ENSP05')]


def test_translation_xrefs_values(dump: CoreDump) -> None:
    row = translation_xrefs(dump).row(0, named=True)
    assert row == {
        'transcriptId': 'ENST02',
        'translationId': 'ENSP02',
        'uniprot_swissprot': ['P00001', 'P00002'],
        'uniprot_trembl': ['A0A001'],
        'uniprot_isoform': ['P00001-2'],
        'alphafold': ['AF-P00001-F1'],
        'signalp': ['SignalP-noTM'],
    }


def test_translation_without_xrefs_has_empty_lists(dump: CoreDump) -> None:
    row = translation_xrefs(dump).row(1, named=True)
    assert row['translationId'] == 'ENSP05'
    fields = ['uniprot_swissprot', 'uniprot_trembl', 'uniprot_isoform', 'alphafold', 'signalp']
    assert all(row[field] == [] for field in fields)


def test_transcript_without_translation_has_no_xref_row(dump: CoreDump) -> None:
    assert 'ENST01' not in translation_xrefs(dump)['transcriptId'].to_list()


def test_translation_on_a_scaffold_gene_has_no_xref_row(dump: CoreDump) -> None:
    assert 'ENSP04' not in translation_xrefs(dump)['translationId'].to_list()


def test_xref_on_a_transcript_is_ignored(dump: CoreDump) -> None:
    assert 'Q00009' not in translation_xrefs(dump)['uniprot_swissprot'].explode().to_list()
