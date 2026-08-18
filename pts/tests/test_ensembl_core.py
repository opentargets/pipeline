"""Tests for the Ensembl core dump reader."""

from __future__ import annotations

import gzip
from pathlib import Path

import pytest

from pts.ensembl_core import CoreDump, CoreDumpError

DDL = """\
CREATE TABLE `gene` (
  `gene_id` int(10) unsigned NOT NULL AUTO_INCREMENT,
  `biotype` varchar(40) NOT NULL,
  `description` text,
  `stable_id` varchar(128) DEFAULT NULL,
  PRIMARY KEY (`gene_id`),
  KEY `stable_id_idx` (`stable_id`)
) ENGINE=MyISAM DEFAULT CHARSET=latin1;
CREATE TABLE `seq_region` (
  `seq_region_id` int(10) unsigned NOT NULL AUTO_INCREMENT,
  `name` varchar(255) NOT NULL,
  PRIMARY KEY (`seq_region_id`)
) ENGINE=MyISAM DEFAULT CHARSET=latin1;
"""


def write_gz(path: Path, text: str) -> None:
    """Write `text` to `path` as gzip, the way the FTP dump ships it."""
    with gzip.open(path, 'wt', encoding='utf8') as handle:
        handle.write(text)


@pytest.fixture
def dump_dir(tmp_path: Path) -> Path:
    """A dump directory carrying only the schema file."""
    write_gz(tmp_path / 'homo_sapiens_core_115_38.sql.gz', DDL)
    return tmp_path


def test_columns_come_from_the_ddl_in_order(dump_dir: Path) -> None:
    assert CoreDump(str(dump_dir)).columns('gene') == ['gene_id', 'biotype', 'description', 'stable_id']


def test_index_lines_are_not_columns(dump_dir: Path) -> None:
    assert 'stable_id_idx' not in CoreDump(str(dump_dir)).columns('gene')


def test_release_comes_from_the_schema_filename(dump_dir: Path) -> None:
    assert CoreDump(str(dump_dir)).release == '115'


def test_unknown_table_raises_and_names_what_is_there(dump_dir: Path) -> None:
    with pytest.raises(CoreDumpError, match='seq_region'):
        CoreDump(str(dump_dir)).columns('nonesuch')


def test_missing_schema_file_raises(tmp_path: Path) -> None:
    with pytest.raises(CoreDumpError, match=r'no .* schema file'):
        CoreDump(str(tmp_path)).columns('gene')
