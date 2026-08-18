"""Tests for the Ensembl core dump reader."""

from __future__ import annotations

import gzip
from pathlib import Path

import polars as pl
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


def test_multiple_schema_files_raise(dump_dir: Path) -> None:
    write_gz(dump_dir / 'homo_sapiens_core_116_39.sql.gz', DDL)
    with pytest.raises(CoreDumpError, match=r'2 .* schema files'):
        CoreDump(str(dump_dir)).columns('gene')


@pytest.fixture
def gene_dump(dump_dir: Path) -> Path:
    """A dump directory with a `gene` table exercising nulls, escapes and quotes."""
    rows = [
        '1\tprotein_coding\ta description\tENSG01',
        # \N is a null; the description is absent
        '2\tlncRNA\t\\N\tENSG02',
        # a literal backslash in the data is written doubled
        '3\tprotein_coding\tUBC4\\\\/5 homolog\tENSG03',
        # a literal backslash followed by the letter t -- NOT a tab
        '4\tprotein_coding\tpath\\\\to\tENSG04',
        # an escaped tab, which is a real tab in the data
        '5\tprotein_coding\tleft\\tright\tENSG05',
        # an unbalanced double quote, which must not start a quoted field
        '6\tprotein_coding\t5" fragment\tENSG06',
    ]
    write_gz(dump_dir / 'gene.txt.gz', '\n'.join(rows) + '\n')
    return dump_dir


def _descriptions(dump_dir: Path) -> list[str | None]:
    frame = CoreDump(str(dump_dir)).scan('gene', {'stable_id': pl.String, 'description': pl.String}).collect()
    return frame['description'].to_list()


def test_scan_returns_only_the_requested_columns(gene_dump: Path) -> None:
    frame = CoreDump(str(gene_dump)).scan('gene', {'gene_id': pl.Int64, 'stable_id': pl.String}).collect()
    assert frame.columns == ['gene_id', 'stable_id']
    assert frame['gene_id'].to_list() == [1, 2, 3, 4, 5, 6]


def test_scan_applies_the_requested_dtypes(gene_dump: Path) -> None:
    frame = CoreDump(str(gene_dump)).scan('gene', {'gene_id': pl.Int64}).collect()
    assert frame.schema['gene_id'] == pl.Int64


def test_null_token_becomes_null(gene_dump: Path) -> None:
    assert _descriptions(gene_dump)[1] is None


def test_doubled_backslash_becomes_one(gene_dump: Path) -> None:
    assert _descriptions(gene_dump)[2] == 'UBC4\\/5 homolog'


def test_escaped_backslash_before_t_does_not_become_a_tab(gene_dump: Path) -> None:
    assert _descriptions(gene_dump)[3] == 'path\\to'


def test_escaped_tab_becomes_a_tab(gene_dump: Path) -> None:
    assert _descriptions(gene_dump)[4] == 'left\tright'


def test_unbalanced_quote_is_data_not_quoting(gene_dump: Path) -> None:
    assert _descriptions(gene_dump)[5] == '5" fragment'


def test_requesting_an_unknown_column_raises(gene_dump: Path) -> None:
    with pytest.raises(CoreDumpError, match='nonesuch'):
        CoreDump(str(gene_dump)).scan('gene', {'nonesuch': pl.String}).collect()
