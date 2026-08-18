"""Tests for the ensembl transformer."""

from __future__ import annotations

import gzip
from pathlib import Path

import polars as pl
import pytest

from pts.ensembl_core import CoreDump
from pts.transformers.ensembl import _genes

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
XREFS = ['10\t1\tHGNC:1\tGENEA\tDIRECT', '11\t1\tHGNC:3\tGENEC\tDIRECT', '12\t1\tHGNC:4\tMTGENE\tDIRECT']


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
