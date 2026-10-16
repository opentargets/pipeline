"""Tests for the shared Ensembl core dump test helpers."""

from __future__ import annotations

import gzip
from pathlib import Path

import polars as pl

from pts.ensembl import CoreDump
from tests.ensembl_fixtures import cells, ddl, write_dump


def test_write_dump_writes_the_schema_and_one_file_per_table(tmp_path: Path) -> None:
    schema = ddl({'gene': ['gene_id', 'stable_id']})
    source = write_dump(tmp_path, {'gene': [['1', 'ENSG01']]}, schema)
    assert source == str(tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ['gene.txt.gz', 'homo_sapiens_core_116_38.sql.gz']
    with gzip.open(tmp_path / 'homo_sapiens_core_116_38.sql.gz', 'rt') as handle:
        assert handle.read() == schema
    with gzip.open(tmp_path / 'gene.txt.gz', 'rt') as handle:
        assert handle.read() == '1\tENSG01\n'


def test_write_dump_round_trips_through_core_dump(tmp_path: Path) -> None:
    schema = ddl({'gene': ['gene_id', 'description', 'stable_id']})
    rows = [cells(1, 'a', 'ENSG01'), cells(2, None, 'ENSG02')]
    dump = CoreDump(write_dump(tmp_path, {'gene': rows}, schema))
    assert dump.release == '116'
    assert dump.columns('gene') == ['gene_id', 'description', 'stable_id']
    frame = dump.scan('gene', {'gene_id': pl.Int64(), 'description': pl.String()}).collect()
    assert frame.rows() == [(1, 'a'), (2, None)]


def test_write_dump_names_the_schema_file_after_db(tmp_path: Path) -> None:
    schema = ddl({'gene': ['gene_id']})
    dump = CoreDump(write_dump(tmp_path, {}, schema, db='mus_musculus_core_116_39'))
    assert (tmp_path / 'mus_musculus_core_116_39.sql.gz').exists()
    assert dump.release == '116'


def test_cells_writes_none_as_the_null_token() -> None:
    assert cells(1, None, 'x', -1) == ['1', '\\N', 'x', '-1']
