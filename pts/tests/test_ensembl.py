"""Tests for the ensembl transformer."""

from __future__ import annotations

import gzip
from pathlib import Path

import polars as pl
import pytest

from pts.ensembl_core import CoreDump
from pts.transformers.ensembl import _genes, _translation_values

SCHEMA = """\
CREATE TABLE `gene` (
  `gene_id` int(10) unsigned NOT NULL AUTO_INCREMENT,
  `biotype` varchar(40) NOT NULL,
  `seq_region_id` int(10) unsigned NOT NULL,
  `seq_region_start` int(10) unsigned NOT NULL,
  `seq_region_end` int(10) unsigned NOT NULL,
  `seq_region_strand` tinyint(2) NOT NULL,
  `display_xref_id` int(10) unsigned DEFAULT NULL,
  `description` text,
  `canonical_transcript_id` int(10) unsigned NOT NULL,
  `stable_id` varchar(128) DEFAULT NULL,
  PRIMARY KEY (`gene_id`)
) ENGINE=MyISAM DEFAULT CHARSET=latin1;
CREATE TABLE `seq_region` (
  `seq_region_id` int(10) unsigned NOT NULL AUTO_INCREMENT,
  `name` varchar(255) NOT NULL,
  `coord_system_id` int(10) unsigned NOT NULL,
  PRIMARY KEY (`seq_region_id`)
) ENGINE=MyISAM DEFAULT CHARSET=latin1;
CREATE TABLE `coord_system` (
  `coord_system_id` int(10) unsigned NOT NULL AUTO_INCREMENT,
  `species_id` int(10) unsigned NOT NULL,
  `name` varchar(40) NOT NULL,
  PRIMARY KEY (`coord_system_id`)
) ENGINE=MyISAM DEFAULT CHARSET=latin1;
CREATE TABLE `xref` (
  `xref_id` int(10) unsigned NOT NULL AUTO_INCREMENT,
  `external_db_id` int(10) unsigned NOT NULL,
  `dbprimary_acc` varchar(512) NOT NULL,
  `display_label` varchar(512) NOT NULL,
  `info_type` varchar(40) NOT NULL,
  PRIMARY KEY (`xref_id`)
) ENGINE=MyISAM DEFAULT CHARSET=latin1;
CREATE TABLE `translation` (
  `translation_id` int(10) unsigned NOT NULL AUTO_INCREMENT,
  `transcript_id` int(10) unsigned NOT NULL,
  `stable_id` varchar(128) DEFAULT NULL,
  PRIMARY KEY (`translation_id`)
) ENGINE=MyISAM DEFAULT CHARSET=latin1;
CREATE TABLE `object_xref` (
  `ensembl_id` int(10) unsigned NOT NULL,
  `ensembl_object_type` varchar(30) NOT NULL,
  `xref_id` int(10) unsigned NOT NULL
) ENGINE=MyISAM DEFAULT CHARSET=latin1;
CREATE TABLE `external_db` (
  `external_db_id` int(10) unsigned NOT NULL AUTO_INCREMENT,
  `db_name` varchar(100) NOT NULL,
  PRIMARY KEY (`external_db_id`)
) ENGINE=MyISAM DEFAULT CHARSET=latin1;
CREATE TABLE `protein_feature` (
  `translation_id` int(10) unsigned NOT NULL,
  `hit_name` varchar(40) NOT NULL,
  `analysis_id` smallint(5) unsigned NOT NULL
) ENGINE=MyISAM DEFAULT CHARSET=latin1;
CREATE TABLE `analysis` (
  `analysis_id` smallint(5) unsigned NOT NULL AUTO_INCREMENT,
  `logic_name` varchar(128) NOT NULL,
  PRIMARY KEY (`analysis_id`)
) ENGINE=MyISAM DEFAULT CHARSET=latin1;
"""

# gene_id, biotype, seq_region_id, start, end, strand, display_xref_id, description, canonical, stable_id
GENES = [
    '1\tprotein_coding\t1\t100\t200\t1\t10\ta gene\t1001\tENSG01',
    '2\tlncRNA\t1\t300\t400\t-1\t\\N\t\\N\t1002\tENSG02',
    # on a scaffold, must be filtered out
    '3\tprotein_coding\t2\t10\t20\t1\t11\ta scaffold gene\t1003\tENSG03',
    '4\tMt_tRNA\t3\t5\t60\t1\t12\ta mito gene\t1004\tENSG04',
]
SEQ_REGIONS = ['1\t1\t1', '2\tKI270728.1\t2', '3\tMT\t1']
COORD_SYSTEMS = ['1\t1\tchromosome', '2\t1\tscaffold']

# xref_id, external_db_id, dbprimary_acc, display_label, info_type
XREFS = [
    '10\t1\tHGNC:1\tGENEA\tDIRECT',
    '11\t1\tHGNC:3\tGENEC\tDIRECT',
    '12\t1\tHGNC:4\tMTGENE\tDIRECT',
    # translation 2001's two swissprot accessions
    '101\t2200\tP00002\tP00002\tDIRECT',
    '102\t2200\tP00001\tP00001\tSEQUENCE_MATCH',
    # translation 2002's two trembl accessions
    '103\t2000\tA0A002\tA0A002\tDIRECT',
    '104\t2000\tA0A001\tA0A001\tSEQUENCE_MATCH',
]

# translation_id, transcript_id, stable_id
TRANSLATIONS = [
    '2001\t1001\tENSP01',
    '2002\t1002\tENSP02',
    # no xrefs and no protein features: every array column must come out null
    '2003\t1003\tENSP03',
]

# ensembl_id, ensembl_object_type, xref_id
OBJECT_XREFS = [
    '2001\tTranslation\t101',
    '2001\tTranslation\t102',
    '2002\tTranslation\t103',
    '2002\tTranslation\t104',
]

# external_db_id, db_name
EXTERNAL_DBS = ['2200\tUniprot/SWISSPROT', '2000\tUniprot/SPTREMBL']

# translation_id, hit_name, analysis_id
PROTEIN_FEATURES = ['2001\tSignalP-noTM\t300']

# analysis_id, logic_name
ANALYSES = ['300\tsignalp']


def write_gz(path: Path, text: str) -> None:
    with gzip.open(path, 'wt', encoding='utf8') as handle:
        handle.write(text)


@pytest.fixture
def dump(tmp_path: Path) -> CoreDump:
    write_gz(tmp_path / 'homo_sapiens_core_115_38.sql.gz', SCHEMA)
    write_gz(tmp_path / 'gene.txt.gz', '\n'.join(GENES) + '\n')
    write_gz(tmp_path / 'seq_region.txt.gz', '\n'.join(SEQ_REGIONS) + '\n')
    write_gz(tmp_path / 'coord_system.txt.gz', '\n'.join(COORD_SYSTEMS) + '\n')
    write_gz(tmp_path / 'xref.txt.gz', '\n'.join(XREFS) + '\n')
    write_gz(tmp_path / 'translation.txt.gz', '\n'.join(TRANSLATIONS) + '\n')
    write_gz(tmp_path / 'object_xref.txt.gz', '\n'.join(OBJECT_XREFS) + '\n')
    write_gz(tmp_path / 'external_db.txt.gz', '\n'.join(EXTERNAL_DBS) + '\n')
    write_gz(tmp_path / 'protein_feature.txt.gz', '\n'.join(PROTEIN_FEATURES) + '\n')
    write_gz(tmp_path / 'analysis.txt.gz', '\n'.join(ANALYSES) + '\n')
    return CoreDump(str(tmp_path))


def test_scaffold_genes_are_filtered_out(dump: CoreDump) -> None:
    assert _genes(dump).collect()['id'].to_list() == ['ENSG01', 'ENSG02', 'ENSG04']


def test_mitochondrial_genes_are_kept(dump: CoreDump) -> None:
    frame = _genes(dump).collect()
    assert frame.filter(pl.col('id') == 'ENSG04')['chromosome'].item() == 'MT'


def test_approved_symbol_comes_from_the_display_xref(dump: CoreDump) -> None:
    frame = _genes(dump).collect()
    assert frame.filter(pl.col('id') == 'ENSG01')['approvedSymbol'].item() == 'GENEA'


def test_a_gene_without_a_display_xref_has_a_null_symbol(dump: CoreDump) -> None:
    frame = _genes(dump).collect()
    assert frame.filter(pl.col('id') == 'ENSG02')['approvedSymbol'].item() is None


def test_coordinates_are_integers_and_strand_is_signed(dump: CoreDump) -> None:
    frame = _genes(dump).collect()
    row = frame.filter(pl.col('id') == 'ENSG02')
    assert row['start'].item() == 300
    assert row['end'].item() == 400
    assert row['strand'].item() == -1


def test_all_accessions_are_kept_not_just_the_first(dump: CoreDump) -> None:
    frame = _translation_values(dump).collect()
    row = frame.filter(pl.col('translationId') == 'ENSP01')
    assert sorted(row['uniprot_swissprot'].item().to_list()) == ['P00001', 'P00002']


def test_accession_arrays_are_sorted(dump: CoreDump) -> None:
    frame = _translation_values(dump).collect()
    assert frame.filter(pl.col('translationId') == 'ENSP02')['uniprot_trembl'].item().to_list() == [
        'A0A001',
        'A0A002',
    ]


def test_signalp_comes_from_the_protein_feature_not_an_xref(dump: CoreDump) -> None:
    frame = _translation_values(dump).collect()
    assert frame.filter(pl.col('translationId') == 'ENSP01')['signalp'].item().to_list() == ['SignalP-noTM']


def test_a_translation_with_no_xrefs_has_null_arrays(dump: CoreDump) -> None:
    frame = _translation_values(dump).collect()
    row = frame.filter(pl.col('translationId') == 'ENSP03')
    assert row['uniprot_swissprot'].item() is None
