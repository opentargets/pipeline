"""Tests for the transcript fields the variant step takes from the Ensembl core tables."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from pts.transformers.variant.index_rows import uniprot_accessions
from pts.transformers.variant.transcripts import canonical_transcript_ids, transcript_fields
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
            (7, 'artifact', 131557, 9500, 9600, 1, None, None, 700, 'ENSG07'),
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
            (700, 7, 131557, 9500, 9600, 1, 'artifact', None, 'ENST_ARTIFACT'),
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
            (700, 554, '1'),
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
            (603, 2000, 'B0B002', 'B0B002'),
            (604, 2200, 'O00002', 'O00002'),
        ],
    ),
    'object_xref': (
        ['object_xref_id', 'ensembl_id', 'ensembl_object_type', 'xref_id'],
        # In neither accession nor id order.
        [
            (5, 1000, 'Translation', 603),
            (4, 1000, 'Translation', 604),
            (1, 1000, 'Translation', 602),
            (2, 1000, 'Translation', 601),
        ],
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
    assert row['swissprot'] == ['O00002', 'P00001']
    assert row['trembl'] == ['B0B002', 'A0A001']


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


def test_artifact_transcripts_absent(core: str) -> None:
    assert 'ENST_ARTIFACT' not in transcript_fields(core, '22')['transcript_id'].to_list()


def test_only_the_requested_chromosome(core: str) -> None:
    assert transcript_fields(core, '1')['transcript_id'].to_list() == ['ENST_CHR1']


def test_canonical_transcript_ids(core: str) -> None:
    assert canonical_transcript_ids(core) == {'ENST_CODING'}


#: The UniProt rows of seven release 116 translations, in the dump's `object_xref` line order, and
#: the accessions VEP 116 reported for their transcripts on do/platform-2610-2.
RELEASE_116_UNIPROT: dict[str, tuple[list[str], list[tuple[object, ...]]]] = {
    'coord_system': (['coord_system_id', 'species_id', 'name'], [(1, 1, 'chromosome')]),
    'seq_region': (['seq_region_id', 'name', 'coord_system_id', 'length'], [(1, '1', 1, 1000)]),
    'gene': (
        TABLES['gene'][0],
        [
            (1, 'protein_coding', 1, 1, 100, 1, None, None, 650920, 'ENSG01'),
            (2, 'protein_coding', 1, 1, 100, 1, None, None, 626524, 'ENSG02'),
            (3, 'protein_coding', 1, 1, 100, 1, None, None, 671030, 'ENSG03'),
            (4, 'protein_coding', 1, 1, 100, 1, None, None, 671415, 'ENSG04'),
            (5, 'protein_coding', 1, 1, 100, 1, None, None, 532843, 'ENSG05'),
            (6, 'protein_coding', 1, 1, 100, 1, None, None, 877542, 'ENSG06'),
            (7, 'protein_coding', 1, 1, 100, 1, None, None, 463933, 'ENSG07'),
        ],
    ),
    'transcript': (
        TABLES['transcript'][0],
        [
            (650920, 1, 1, 1, 100, 1, 'protein_coding', None, 'ENST00000229829'),
            (626524, 2, 1, 1, 100, 1, 'protein_coding', None, 'ENST00000369134'),
            (671030, 3, 1, 1, 100, 1, 'protein_coding', None, 'ENST00000375864'),
            (671415, 4, 1, 1, 100, 1, 'protein_coding', None, 'ENST00000440137'),
            (532843, 5, 1, 1, 100, 1, 'protein_coding', None, 'ENST00000445148'),
            (877542, 6, 1, 1, 100, 1, 'protein_coding', None, 'ENST00000532726'),
            (463933, 7, 1, 1, 100, 1, 'protein_coding', None, 'ENST00000541038'),
        ],
    ),
    'translation': (
        ['translation_id', 'transcript_id', 'stable_id'],
        [
            (1211214, 650920, 'ENSP01'),
            (1190267, 626524, 'ENSP02'),
            (1225800, 671030, 'ENSP03'),
            (1226050, 671415, 'ENSP04'),
            (1127402, 532843, 'ENSP05'),
            (1389407, 877542, 'ENSP06'),
            (1102337, 463933, 'ENSP07'),
        ],
    ),
    'exon': (TABLES['exon'][0], []),
    'exon_transcript': (TABLES['exon_transcript'][0], []),
    'attrib_type': (TABLES['attrib_type'][0], []),
    'transcript_attrib': (TABLES['transcript_attrib'][0], []),
    'external_db': (['external_db_id', 'db_name'], [(2000, 'Uniprot/SPTREMBL'), (2200, 'Uniprot/SWISSPROT')]),
    'xref': (
        ['xref_id', 'external_db_id', 'dbprimary_acc', 'display_label'],
        [
            (585532, 2000, 'B2R986', 'B2R986'),
            (795726, 2000, 'X5CF87', 'X5CF87'),
            (796398, 2000, 'A9ZM15', 'A9ZM15'),
            (846155, 2200, 'P06340', 'P06340'),
            (1006277, 2200, 'O76087', 'O76087'),
            (1029201, 2200, 'P0CL80', 'P0CL80'),
            (1029232, 2200, 'P0CL81', 'P0CL81'),
            (1029268, 2200, 'P0CL82', 'P0CL82'),
            (1041116, 2200, 'Q969I3', 'Q969I3'),
            (1064270, 2200, 'Q13002', 'Q13002'),
            (1244359, 2000, 'A0A1V0E3Q8', 'A0A1V0E3Q8'),
            (1349315, 2200, 'Q8NDX9', 'Q8NDX9'),
            (1363147, 2000, 'A0A1U9X7Y3', 'A0A1U9X7Y3'),
            (1575741, 2000, 'A0A1V0E3R3', 'A0A1V0E3R3'),
            (1619890, 2000, 'A0A1V0E3M7', 'A0A1V0E3M7'),
            (1640966, 2000, 'F5H6U5', 'F5H6U5'),
            (1655644, 2200, 'Q96P68', 'Q96P68'),
            (1686290, 2000, 'A0A1V0E3T1', 'A0A1V0E3T1'),
            (1709885, 2000, 'A0A1V0E3Q2', 'A0A1V0E3Q2'),
            (1722052, 2000, 'A0A1V0E3P6', 'A0A1V0E3P6'),
            (1754214, 2000, 'A0A1V0E3S6', 'A0A1V0E3S6'),
            (1798984, 2000, 'A0A1V0E3P8', 'A0A1V0E3P8'),
            (1828380, 2000, 'A0A1V0E3P3', 'A0A1V0E3P3'),
            (1834891, 2000, 'A0A1V0E3S7', 'A0A1V0E3S7'),
            (1835341, 2000, 'N0E641', 'N0E641'),
            (1838303, 2000, 'A0A1V0E3P9', 'A0A1V0E3P9'),
            (1839511, 2000, 'A0A1V0E3P1', 'A0A1V0E3P1'),
            (1846377, 2000, 'A0A1V0E3Q6', 'A0A1V0E3Q6'),
            (1849588, 2000, 'A0A1V0E3R6', 'A0A1V0E3R6'),
            (1881025, 2000, 'A0A1V0E3N7', 'A0A1V0E3N7'),
            (1897761, 2000, 'E9PR27', 'E9PR27'),
            (1909570, 2000, 'A0A1V0E3S0', 'A0A1V0E3S0'),
            (1918246, 2000, 'A0A1V0E3Q1', 'A0A1V0E3Q1'),
            (1942059, 2000, 'A0A1V0E3N3', 'A0A1V0E3N3'),
            (1955672, 2000, 'A0A1V0E3Q4', 'A0A1V0E3Q4'),
            (1960320, 2000, 'A0A1V0E3N6', 'A0A1V0E3N6'),
            (1970465, 2000, 'A0A1V0E3P0', 'A0A1V0E3P0'),
            (1986168, 2000, 'F8WEZ8', 'F8WEZ8'),
            (1987554, 2000, 'A0A1V0E3R8', 'A0A1V0E3R8'),
            (1990900, 2000, 'A0A1V0E3R0', 'A0A1V0E3R0'),
            (1994939, 2000, 'A0A1V0E3R4', 'A0A1V0E3R4'),
            (2009594, 2000, 'A0A1V0E3Q3', 'A0A1V0E3Q3'),
            (2029683, 2000, 'A0A8D9PH75', 'A0A8D9PH75'),
            (2035967, 2000, 'A0A1V0E3S4', 'A0A1V0E3S4'),
            (2056945, 2000, 'A0A1V0E3N1', 'A0A1V0E3N1'),
            (2067774, 2000, 'A0A1V0E3Q7', 'A0A1V0E3Q7'),
            (2101065, 2000, 'A0A1V0E3Q5', 'A0A1V0E3Q5'),
        ],
    ),
    'object_xref': (
        ['object_xref_id', 'ensembl_id', 'ensembl_object_type', 'xref_id'],
        [
            (3986219, 1102337, 'Translation', 1640966),
            (4110541, 1389407, 'Translation', 1897761),
            (4120962, 1190267, 'Translation', 1986168),
            (14701214, 1102337, 'Translation', 585532),
            (13634564, 1389407, 'Translation', 796398),
            (15678452, 1225800, 'Translation', 1835341),
            (11901791, 1190267, 'Translation', 2029683),
            (7068962, 1211214, 'Translation', 795726),
            (11740934, 1211214, 'Translation', 1244359),
            (6342044, 1225800, 'Translation', 1363147),
            (6764190, 1211214, 'Translation', 1575741),
            (11644677, 1211214, 'Translation', 1619890),
            (12808140, 1211214, 'Translation', 1686290),
            (13920717, 1211214, 'Translation', 1709885),
            (9159074, 1211214, 'Translation', 1722052),
            (12201245, 1211214, 'Translation', 1754214),
            (12654135, 1211214, 'Translation', 1798984),
            (13000139, 1211214, 'Translation', 1828380),
            (11526300, 1211214, 'Translation', 1834891),
            (12765253, 1211214, 'Translation', 1838303),
            (12768375, 1211214, 'Translation', 1839511),
            (7704137, 1211214, 'Translation', 1846377),
            (7714989, 1211214, 'Translation', 1849588),
            (13569354, 1211214, 'Translation', 1881025),
            (17411188, 1211214, 'Translation', 1909570),
            (14911116, 1211214, 'Translation', 1918246),
            (8921984, 1211214, 'Translation', 1942059),
            (9509030, 1211214, 'Translation', 1955672),
            (12004765, 1211214, 'Translation', 1960320),
            (15278850, 1211214, 'Translation', 1970465),
            (14176711, 1211214, 'Translation', 1987554),
            (14097712, 1211214, 'Translation', 1990900),
            (13570405, 1211214, 'Translation', 1994939),
            (16851599, 1211214, 'Translation', 2009594),
            (16621282, 1211214, 'Translation', 2035967),
            (15369024, 1211214, 'Translation', 2056945),
            (15375975, 1211214, 'Translation', 2067774),
            (16448270, 1211214, 'Translation', 2101065),
            (3057846, 1211214, 'Translation', 846155),
            (3205948, 1226050, 'Translation', 1029201),
            (3205967, 1127402, 'Translation', 1029232),
            (3214633, 1389407, 'Translation', 1041116),
            (3233684, 1190267, 'Translation', 1064270),
            (3505602, 1225800, 'Translation', 1349315),
            (3990510, 1102337, 'Translation', 1655644),
            (7859498, 1127402, 'Translation', 1006277),
            (7859550, 1226050, 'Translation', 1006277),
            (6212871, 1127402, 'Translation', 1029268),
            (6212889, 1226050, 'Translation', 1029268),
        ],
    ),
}
VEP_116_UNIPROT = {
    'ENST00000229829': [
        'P06340', 'X5CF87', 'A0A1V0E3Q8', 'A0A1V0E3R3', 'A0A1V0E3M7', 'A0A1V0E3T1', 'A0A1V0E3Q2', 'A0A1V0E3P6',
        'A0A1V0E3S6', 'A0A1V0E3P8', 'A0A1V0E3P3', 'A0A1V0E3S7', 'A0A1V0E3P9', 'A0A1V0E3P1', 'A0A1V0E3Q6',
        'A0A1V0E3R6', 'A0A1V0E3N7', 'A0A1V0E3S0', 'A0A1V0E3Q1', 'A0A1V0E3N3', 'A0A1V0E3Q4', 'A0A1V0E3N6',
        'A0A1V0E3P0', 'A0A1V0E3R8', 'A0A1V0E3R0', 'A0A1V0E3R4', 'A0A1V0E3Q3', 'A0A1V0E3S4', 'A0A1V0E3N1',
        'A0A1V0E3Q7', 'A0A1V0E3Q5',
    ],
    'ENST00000369134': ['Q13002', 'F8WEZ8', 'A0A8D9PH75'],
    'ENST00000375864': ['Q8NDX9', 'N0E641', 'A0A1U9X7Y3'],
    'ENST00000440137': ['P0CL80', 'O76087', 'P0CL82'],
    'ENST00000445148': ['P0CL81', 'O76087', 'P0CL82'],
    'ENST00000532726': ['Q969I3', 'E9PR27', 'A9ZM15'],
    'ENST00000541038': ['Q96P68', 'F5H6U5', 'B2R986'],
}  # fmt: skip


def test_uniprot_accessions_in_vep_order(tmp_path: Path) -> None:
    schema_ddl = ddl({name: columns for name, (columns, _) in RELEASE_116_UNIPROT.items()})
    rows = {name: [cells(*row) for row in table_rows] for name, (_, table_rows) in RELEASE_116_UNIPROT.items()}
    frame = transcript_fields(write_dump(tmp_path, rows, schema_ddl), '1').select(
        'transcript_id', uniprot_accessions(pl.col('swissprot'), pl.col('trembl')).alias('accessions')
    )

    assert dict(frame.iter_rows()) == VEP_116_UNIPROT
