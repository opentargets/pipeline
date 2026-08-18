"""Tests for the ephemeral postgres helper."""

import subprocess
import tarfile
import zipfile
from collections.abc import Iterator
from pathlib import Path

import polars as pl
import pytest
from pixeltable_pgserver.postgres_server import get_server

from pts.postgres import (
    DUMP_MAGIC,
    MAX_ARCHIVE_VERSION,
    PostgresError,
    _build_restore_args,
    _build_select_sql,
    _check_archive_version,
    _resolve_archive_member,
    read_dump_tables,
    restored_dump,
)


class TestBuildSelectSql:
    def test_golden(self) -> None:
        assert _build_select_sql('studies', 'ctgov', ['nct_id', 'phase']) == (
            'SELECT DISTINCT "nct_id", "phase" FROM "ctgov"."studies"'
        )

    def test_all_columns(self) -> None:
        assert _build_select_sql('t', 'public') == 'SELECT DISTINCT * FROM "public"."t"'

    def test_always_deduplicates(self) -> None:
        """Row counts downstream were all measured with duplicates removed."""
        assert _build_select_sql('t', 'public', ['id']).startswith('SELECT DISTINCT ')


class TestBuildRestoreArgs:
    @pytest.fixture
    def passes(self) -> list[list[str]]:
        return _build_restore_args(Path('/bin'), 'postgresql://x', Path('/dump.dmp'), ['studies', 'conditions'], 8)

    def test_two_passes(self, passes: list[list[str]]) -> None:
        assert len(passes) == 2

    def test_first_pass_restores_the_whole_schema(self, passes: list[list[str]]) -> None:
        pre_data, _ = passes
        assert '--section=pre-data' in pre_data
        assert '--table' not in pre_data

    def test_second_pass_restores_only_the_requested_tables(self, passes: list[list[str]]) -> None:
        _, data = passes
        assert '--section=data' in data
        assert data.count('--table') == 2
        assert 'studies' in data
        assert 'conditions' in data
        assert data[data.index('--jobs') + 1] == '8'

    def test_post_data_is_never_restored(self, passes: list[list[str]]) -> None:
        assert not any('post-data' in arg for p in passes for arg in p)

    def test_ownership_and_privileges_are_skipped(self, passes: list[list[str]]) -> None:
        for p in passes:
            assert '--no-owner' in p
            assert '--no-privileges' in p

    def test_dump_is_the_last_argument(self, passes: list[list[str]]) -> None:
        for p in passes:
            assert p[-1] == '/dump.dmp'


class TestResolveArchiveMember:
    def test_exact_name(self) -> None:
        assert _resolve_archive_member(['a.txt', 'postgres.dmp'], 'postgres.dmp') == 'postgres.dmp'

    def test_glob(self) -> None:
        names = ['chembl_37/INSTALL_postgresql', 'chembl_37/chembl_37_postgresql.dmp']
        assert _resolve_archive_member(names, '*.dmp') == 'chembl_37/chembl_37_postgresql.dmp'

    def test_no_match(self) -> None:
        with pytest.raises(PostgresError, match='matched 0 members'):
            _resolve_archive_member(['a.txt'], '*.dmp')

    def test_ambiguous_match(self) -> None:
        with pytest.raises(PostgresError, match='matched 2 members'):
            _resolve_archive_member(['a.dmp', 'b.dmp'], '*.dmp')


class TestCheckArchiveVersion:
    def _write(self, path: Path, header: bytes) -> Path:
        path.write_bytes(header + b'\x00' * 32)
        return path

    def test_accepts_the_versions_aact_and_chembl_ship(self, tmp_path: Path) -> None:
        for minor in (13, 14, 16):
            dump = self._write(tmp_path / f'{minor}.dmp', DUMP_MAGIC + bytes([1, minor, 0]))
            _check_archive_version(dump)

    def test_rejects_a_newer_archive(self, tmp_path: Path) -> None:
        newer = MAX_ARCHIVE_VERSION[1] + 1
        dump = self._write(tmp_path / 'new.dmp', DUMP_MAGIC + bytes([1, newer, 0]))
        with pytest.raises(PostgresError, match='bundled postgres needs upgrading'):
            _check_archive_version(dump)

    def test_rejects_something_that_is_not_a_dump(self, tmp_path: Path) -> None:
        dump = self._write(tmp_path / 'nope.zip', b'PK\x03\x04')
        with pytest.raises(PostgresError, match='not a pg_dump archive'):
            _check_archive_version(dump)


@pytest.mark.pgserver
class TestRestoredDump:
    """Restore a real dump into a real server and read it, with nothing mocked.

    This is what catches an embedded-postgres API change, an incompatible dump,
    or polars failing to reach postgres over its unix socket — the server listens
    on nothing else, so every read in the pipeline depends on that working.
    """

    @pytest.fixture(scope='class')
    def dump(self, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
        path = tmp_path_factory.mktemp('source')
        server = get_server(path / 'pgdata', cleanup_mode='delete')
        try:
            server.psql(
                'CREATE SCHEMA demo;'
                'CREATE TABLE demo.t (id int, txt text);'
                "INSERT INTO demo.t SELECT i, 'x' || (i % 7) FROM generate_series(1, 1000) i;"
            )
            dump = path / 'demo.dmp'
            subprocess.run(
                [str(server.bin_path / 'pg_dump'), '-Fc', '-f', str(dump), server.get_uri()],
                check=True,
            )
            yield dump
        finally:
            server.cleanup()

    def _read(self, source: Path, scratch_root: Path) -> pl.DataFrame:
        return read_dump_tables(
            str(source), {'t': ['id', 'txt']}, schema_name='demo', scratch_root=scratch_root
        )['t']

    def test_reads_a_restored_table_with_polars(self, dump: Path, tmp_path: Path) -> None:
        df = self._read(dump, tmp_path)
        assert df.height == 1000
        assert df.columns == ['id', 'txt']
        assert df['id'].n_unique() == 1000

    def test_deduplicates_a_projected_read(self, dump: Path, tmp_path: Path) -> None:
        """``txt`` holds 7 distinct values across the 1000 rows."""
        df = read_dump_tables(str(dump), {'t': ['txt']}, schema_name='demo', scratch_root=tmp_path)['t']
        assert df.height == 7

    def test_an_empty_table_is_an_error(self, tmp_path: Path) -> None:
        """A restore that loads nothing is silent everywhere else."""
        source = tmp_path / 'source'
        source.mkdir()
        server = get_server(source / 'pgdata', cleanup_mode='delete')
        try:
            server.psql('CREATE SCHEMA demo; CREATE TABLE demo.empty (id int);')
            dump = tmp_path / 'empty.dmp'
            subprocess.run(
                [str(server.bin_path / 'pg_dump'), '-Fc', '-f', str(dump), server.get_uri()], check=True
            )
        finally:
            server.cleanup()

        with pytest.raises(PostgresError, match='came back empty'):
            read_dump_tables(str(dump), {'empty': ['id']}, schema_name='demo', scratch_root=tmp_path)

    def test_zip(self, dump: Path, tmp_path: Path) -> None:
        archive = tmp_path / 'demo.zip'
        with zipfile.ZipFile(archive, 'w') as z:
            z.write(dump, 'nested/postgres.dmp')
        assert self._read(archive, tmp_path).height == 1000

    def test_tar_gz(self, dump: Path, tmp_path: Path) -> None:
        archive = tmp_path / 'demo.tar.gz'
        with tarfile.open(archive, 'w:gz') as t:
            t.add(dump, 'chembl/chembl_postgresql.dmp')
        assert self._read(archive, tmp_path).height == 1000

    def test_scratch_is_cleaned_up(self, dump: Path, tmp_path: Path) -> None:
        scratch_root = tmp_path / 'scratch'
        scratch_root.mkdir()
        self._read(dump, scratch_root)
        assert not list(scratch_root.iterdir())

    def test_scratch_is_cleaned_up_when_the_body_raises(self, dump: Path, tmp_path: Path) -> None:
        scratch_root = tmp_path / 'scratch'
        scratch_root.mkdir()
        with (
            pytest.raises(ZeroDivisionError),
            restored_dump(str(dump), tables=['t'], scratch_root=scratch_root),
        ):
            1 / 0  # noqa: B018 the point is to leave the block by raising
        assert not list(scratch_root.iterdir())
