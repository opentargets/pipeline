"""Tests for the Ensembl core dump reader."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from otter.storage.synchronous.google import GoogleStorage

from pts.ensembl import CoreDump, CoreDumpError
from tests.ensembl_fixtures import write_dump, write_gz

DDL = """\
CREATE TABLE `gene` (
  `gene_id` int(10) unsigned NOT NULL AUTO_INCREMENT,
  `biotype` varchar(40) NOT NULL,
  `description` text,
  `stable_id` varchar(128) DEFAULT NULL,
  PRIMARY KEY (`gene_id`),
  KEY `stable_id_idx` (`stable_id`)
) ENGINE=MyISAM AUTO_INCREMENT=156308 DEFAULT CHARSET=latin1 ROW_FORMAT=DYNAMIC;
CREATE TABLE `seq_region` (
  `seq_region_id` int(10) unsigned NOT NULL AUTO_INCREMENT,
  `name` varchar(255) NOT NULL,
  PRIMARY KEY (`seq_region_id`)
) ENGINE=MyISAM DEFAULT CHARSET=latin1;
"""


@pytest.fixture
def dump_dir(tmp_path: Path) -> Path:
    return Path(write_dump(tmp_path, {}, DDL))


def test_columns_come_from_the_ddl_in_order(dump_dir: Path) -> None:
    assert CoreDump(str(dump_dir)).columns('gene') == ['gene_id', 'biotype', 'description', 'stable_id']


def test_index_lines_are_not_columns(dump_dir: Path) -> None:
    assert 'stable_id_idx' not in CoreDump(str(dump_dir)).columns('gene')


def test_release_comes_from_the_schema_filename(dump_dir: Path) -> None:
    assert CoreDump(str(dump_dir)).release == '116'


def test_release_skips_the_division_release_number(tmp_path: Path) -> None:
    write_gz(tmp_path / 'caenorhabditis_elegans_core_63_116_282.sql.gz', DDL)
    assert CoreDump(str(tmp_path)).release == '116'


def test_release_ignores_the_directory_name(tmp_path: Path) -> None:
    misleading = tmp_path / 'archive_core_999_1'
    misleading.mkdir()
    write_gz(misleading / 'homo_sapiens_core_116_38.sql.gz', DDL)
    assert CoreDump(str(misleading)).release == '116'


def test_release_without_a_number_raises(tmp_path: Path) -> None:
    write_gz(tmp_path / 'schema.sql.gz', DDL)
    with pytest.raises(CoreDumpError, match='release number'):
        _ = CoreDump(str(tmp_path)).release


def test_unknown_table_raises_and_names_what_is_there(dump_dir: Path) -> None:
    with pytest.raises(CoreDumpError, match=r"release 116; it has \['gene', 'seq_region'\]"):
        CoreDump(str(dump_dir)).columns('nonesuch')


def test_missing_schema_file_raises(tmp_path: Path) -> None:
    with pytest.raises(CoreDumpError, match=r'no \*\.sql\.gz schema file'):
        CoreDump(str(tmp_path)).columns('gene')


def test_multiple_schema_files_raise(dump_dir: Path) -> None:
    write_gz(dump_dir / 'homo_sapiens_core_115_38.sql.gz', DDL)
    with pytest.raises(CoreDumpError, match=r'2 \*\.sql\.gz schema files'):
        CoreDump(str(dump_dir)).columns('gene')


#: gene_id, biotype, description, stable_id; descriptions as MySQL's outfile writes them.
GENE_ROWS = [
    ['1', 'protein_coding', 'plain', 'ENSG01'],
    ['2', 'lncRNA', '\\N', 'ENSG02'],
    ['3', 'protein_coding', 'UBC4\\\\/5', 'ENSG03'],
    ['4', 'protein_coding', 'path\\\\to', 'ENSG04'],
    ['5', 'protein_coding', 'left\\tright', 'ENSG05'],
    ['6', 'protein_coding', 'line\\nbreak', 'ENSG06'],
    ['7', 'protein_coding', 'carriage\\rreturn', 'ENSG07'],
    ['8', 'protein_coding', 'nul\\0byte', 'ENSG08'],
    ['9', 'protein_coding', 'back\\bspace', 'ENSG09'],
    ['10', 'protein_coding', 'ctrl\\Zz', 'ENSG10'],
    ['11', 'protein_coding', '5" fragment', 'ENSG11'],
    ['12', 'protein_coding', 'trailing\\\\', 'ENSG12'],
    ['13', 'protein_coding', 'slash\\\\\\ttab', 'ENSG13'],
    ['14', 'protein_coding', 'slash\\\\0zero', 'ENSG14'],
    ['15', 'protein_coding', '', 'ENSG15'],
]


@pytest.fixture
def gene_dump(tmp_path: Path) -> Path:
    return Path(write_dump(tmp_path, {'gene': GENE_ROWS}, DDL))


def _description(dump_dir: Path, gene_id: int) -> str | None:
    frame = CoreDump(str(dump_dir)).scan('gene', {'gene_id': pl.Int64(), 'description': pl.String()}).collect()
    return frame.filter(pl.col('gene_id') == gene_id)['description'].item()


@pytest.mark.parametrize(
    ('gene_id', 'expected'),
    [
        (1, 'plain'),
        (3, 'UBC4\\/5'),
        (4, 'path\\to'),
        (5, 'left\tright'),
        (6, 'line\nbreak'),
        (7, 'carriage\rreturn'),
        (8, 'nul\x00byte'),
        (9, 'back\bspace'),
        (10, 'ctrl\x1az'),
        (11, '5" fragment'),
        (12, 'trailing\\'),
        (13, 'slash\\\ttab'),
        (14, 'slash\\0zero'),
        (15, ''),
    ],
)
def test_escapes_are_resolved(gene_dump: Path, gene_id: int, expected: str) -> None:
    assert _description(gene_dump, gene_id) == expected


def test_null_token_becomes_null(gene_dump: Path) -> None:
    assert _description(gene_dump, 2) is None


def test_scan_returns_only_the_requested_columns_in_the_requested_order(gene_dump: Path) -> None:
    frame = CoreDump(str(gene_dump)).scan('gene', {'stable_id': pl.String(), 'gene_id': pl.Int64()}).collect()
    assert frame.columns == ['stable_id', 'gene_id']
    assert frame['gene_id'].to_list() == list(range(1, 16))


def test_scan_applies_the_requested_dtypes(gene_dump: Path) -> None:
    frame = CoreDump(str(gene_dump)).scan('gene', {'gene_id': pl.Int32()}).collect()
    assert frame.schema == pl.Schema({'gene_id': pl.Int32()})


def test_requesting_an_unknown_column_raises(gene_dump: Path) -> None:
    with pytest.raises(CoreDumpError, match=r"gene in release 116 has no column\(s\) \['nonesuch'\]"):
        CoreDump(str(gene_dump)).scan('gene', {'nonesuch': pl.String()})


#: A gs:// prefix served from tmp_path by the patched backend below.
REMOTE = 'gs://test-bucket/release/input/ensembl/homo_sapiens'


@pytest.fixture
def remote_dump(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Serve `REMOTE` from `tmp_path` through otter's real gs:// backend class, and record the polars reads.

    otter's `GoogleStorage.glob` and `.open` are redirected to the local files, and so is the URI handed
    to `pl.scan_csv`. Everything in `CoreDump` between them runs as it does on the VM.
    """
    write_dump(tmp_path, {'gene': GENE_ROWS}, DDL)
    local = lambda uri: str(tmp_path / uri.removeprefix(f'{REMOTE}/'))
    monkeypatch.setattr(
        GoogleStorage,
        'glob',
        lambda self, location, pattern='*': [f'{location}/{p.name}' for p in tmp_path.glob(pattern)],
    )
    monkeypatch.setattr(
        GoogleStorage, 'open', lambda self, location, mode='r': io.BytesIO(Path(local(location)).read_bytes())
    )
    scanned: list[str] = []
    real_scan_csv = pl.scan_csv

    def scan_csv(source: str, **kwargs: Any) -> pl.LazyFrame:
        scanned.append(source)
        return real_scan_csv(local(source), **kwargs)

    monkeypatch.setattr(pl, 'scan_csv', scan_csv)
    return scanned


def test_a_gs_dump_reads_ddl_and_tables_from_gs_uris(remote_dump: list[str]) -> None:
    dump = CoreDump(f'{REMOTE}/')
    assert dump.release == '116'
    assert dump.columns('gene') == ['gene_id', 'biotype', 'description', 'stable_id']
    frame = dump.scan('gene', {'gene_id': pl.Int64(), 'description': pl.String()}).collect()
    assert remote_dump == [f'{REMOTE}/gene.txt.gz']
    assert frame.filter(pl.col('gene_id') == 5)['description'].item() == 'left\tright'
