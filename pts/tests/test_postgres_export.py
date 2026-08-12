import asyncio
import subprocess
import tarfile
import zipfile
from collections.abc import Awaitable, Iterator
from pathlib import Path
from threading import Event

import duckdb
import pytest
from otter.config.model import Config
from otter.manifest.model import Result
from otter.scratchpad.model import Scratchpad
from otter.task.model import Task, TaskContext
from pixeltable_pgserver.postgres_server import get_server
from pydantic import ValidationError

from pts.tasks.postgres_export import (
    DUMP_MAGIC,
    MAX_ARCHIVE_VERSION,
    PostgresExport,
    PostgresExportError,
    PostgresExportSpec,
    TableSpec,
    _build_copy_sql,
    _build_count_sql,
    _build_restore_args,
    _check_archive_version,
    _resolve_archive_member,
    _slug,
)

DESTINATION = 'out/t.parquet'

STUDIES = TableSpec(
    table='studies',
    destination='input/clinical_report/aact/studies.parquet',
    columns=['nct_id', 'overall_status', 'phase'],
)


class TestBuildCopySql:
    def test_golden(self) -> None:
        sql = _build_copy_sql(STUDIES, 'ctgov', Path('/work/studies.parquet'))
        assert sql == (
            'COPY (SELECT DISTINCT "nct_id", "overall_status", "phase" FROM pg."ctgov"."studies") '
            "TO '/work/studies.parquet' (FORMAT parquet, COMPRESSION zstd)"
        )

    def test_all_columns(self) -> None:
        table = TableSpec(table='t', destination='d.parquet')
        assert 'FROM pg."public"."t"' in _build_copy_sql(table, 'public', Path('/d.parquet'))

    def test_always_deduplicates(self) -> None:
        table = TableSpec(table='t', destination='d.parquet')
        assert 'SELECT DISTINCT *' in _build_copy_sql(table, 'public', Path('/d.parquet'))

    def test_where(self) -> None:
        table = TableSpec(table='t', destination='d.parquet', where="phase = 'PHASE3'")
        sql = _build_copy_sql(table, 'public', Path('/d.parquet'))
        assert sql.startswith('COPY (SELECT DISTINCT * FROM pg."public"."t" WHERE phase = \'PHASE3\')')

    def test_quotes_in_destination_are_escaped(self) -> None:
        table = TableSpec(table='t', destination="d's.parquet")
        assert "TO 'd''s.parquet'" in _build_copy_sql(table, 'public', Path("d's.parquet"))


class TestBuildCountSql:
    def test_pushes_the_count_down_to_postgres(self) -> None:
        assert _build_count_sql(STUDIES, 'ctgov') == (
            'SELECT * FROM postgres_query(\'pg\', \'SELECT count(*) FROM "ctgov"."studies"\')'
        )


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


class TestAllowEmpty:
    def test_the_data_pass_uses_strict_names(self) -> None:
        """A --table that matches nothing must fail the restore, not exit 0."""
        _, data = _build_restore_args(Path('/bin'), 'postgresql://x', Path('d.dmp'), ['studies'], 8)
        assert '--strict-names' in data

    def test_count_sql_applies_the_where(self) -> None:
        table = TableSpec(table='studies', destination=DESTINATION, where='phase = 1')
        assert 'WHERE phase = 1' in _build_count_sql(table, 'ctgov')

    def test_allow_empty_defaults_to_false(self) -> None:
        assert TableSpec(table='studies', destination=DESTINATION).allow_empty is False


class TestResolveArchiveMember:
    def test_exact_name(self) -> None:
        assert _resolve_archive_member(['a.txt', 'postgres.dmp'], 'postgres.dmp') == 'postgres.dmp'

    def test_glob(self) -> None:
        names = ['chembl_37/INSTALL_postgresql', 'chembl_37/chembl_37_postgresql.dmp']
        assert _resolve_archive_member(names, '*.dmp') == 'chembl_37/chembl_37_postgresql.dmp'

    def test_no_match(self) -> None:
        with pytest.raises(PostgresExportError, match='matched 0 members'):
            _resolve_archive_member(['a.txt'], '*.dmp')

    def test_ambiguous_match(self) -> None:
        with pytest.raises(PostgresExportError, match='matched 2 members'):
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
        with pytest.raises(PostgresExportError, match='bundled postgres needs upgrading'):
            _check_archive_version(dump)

    def test_rejects_something_that_is_not_a_dump(self, tmp_path: Path) -> None:
        dump = self._write(tmp_path / 'nope.zip', b'PK\x03\x04')
        with pytest.raises(PostgresExportError, match='not a pg_dump archive'):
            _check_archive_version(dump)


class TestSpec:
    def test_requires_at_least_one_table(self) -> None:
        with pytest.raises(ValidationError):
            PostgresExportSpec(name='postgres_export nothing', source='d.dmp', tables=[])

    def test_requires_a_description_in_the_name(self) -> None:
        with pytest.raises(ValidationError):
            PostgresExportSpec(name='postgres_export', source='d.dmp', tables=[STUDIES])

    def test_defaults(self) -> None:
        spec = PostgresExportSpec(name='postgres_export some tables', source='d.dmp', tables=[STUDIES])
        assert spec.task_type == 'postgres_export'
        assert spec.archive_member == '*.dmp'
        assert spec.schema_name == 'public'


class TestSlug:
    def test_is_a_usable_directory_name(self) -> None:
        assert _slug('postgres_export AACT tables!') == 'postgres_export_aact_tables'


def _count(parquet: Path, expression: str = 'count(*)') -> tuple:
    """Read aggregates off a parquet file."""
    sql = f'SELECT {expression} FROM read_parquet(?)'  # noqa: S608 test-local expression
    row = duckdb.connect().execute(sql, [str(parquet)]).fetchone()
    assert row is not None
    return row


def _await(awaitable: Awaitable[Task]) -> Task:
    """Run one of a task's methods.

    ``report`` turns them into awaitables rather than coroutines, which is what
    ``asyncio.run`` wants, so drive a loop directly.
    """
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(awaitable)
    finally:
        loop.close()


class TestResetScratch:
    def _build(self, work_path: Path) -> PostgresExport:
        spec = PostgresExportSpec(
            name='postgres_export demo',
            source='demo.dmp',
            tables=[TableSpec(table='t', destination=DESTINATION, columns=['id'])],
        )
        config = Config(step='demo', steps=['demo'], work_path=work_path, pool_size=2, log_level='DEBUG')
        context = TaskContext(config=config, scratchpad=Scratchpad())
        # the worker assigns this when it hands a task out; here nobody does
        context.abort = Event()
        return PostgresExport(spec, context)

    def test_a_surviving_cluster_is_removed(self, tmp_path: Path) -> None:
        """A killed run leaves pgdata behind; the next attempt must not reuse it."""
        task = self._build(tmp_path / 'work')
        pgdata = task.scratch / 'pgdata'
        pgdata.mkdir(parents=True)
        # PG_VERSION is what makes get_server skip initdb and restart the cluster
        (pgdata / 'PG_VERSION').write_text('16\n')

        task._reset_scratch()

        assert task.scratch.is_dir(), 'the scratch directory itself must still exist'
        assert not pgdata.exists(), 'a surviving cluster would be restarted with its old data'

    def test_it_is_safe_when_nothing_survived(self, tmp_path: Path) -> None:
        task = self._build(tmp_path / 'work')
        task._reset_scratch()
        assert task.scratch.is_dir()


@pytest.mark.pgserver
class TestRoundTrip:
    """Restore a real dump into a real server and export it, with nothing mocked.

    This is what catches an embedded-postgres API change, an incompatible dump, or duckdb
    failing to reach postgres over its unix socket.
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
                'CREATE TABLE demo.empty (id int, txt text);'
            )
            dump = path / 'demo.dmp'
            subprocess.run(
                [str(server.bin_path / 'pg_dump'), '-Fc', '-f', str(dump), server.get_uri()],
                check=True,
            )
            yield dump
        finally:
            server.cleanup()

    def _build(self, work_path: Path, source: str, tables: list[TableSpec] | None = None) -> PostgresExport:
        spec = PostgresExportSpec(
            name='postgres_export demo',
            source=source,
            schema_name='demo',
            tables=tables or [TableSpec(table='t', destination=DESTINATION, columns=['id', 'txt'])],
        )
        config = Config(step='demo', steps=['demo'], work_path=work_path, pool_size=2, log_level='DEBUG')
        context = TaskContext(config=config, scratchpad=Scratchpad())
        # the worker assigns this when it hands a task out; here nobody does
        context.abort = Event()
        return PostgresExport(spec, context)

    def _run(self, work_path: Path, source: str) -> Path:
        task = self._build(work_path, source)
        _await(task.run())
        # report() records exceptions in the manifest instead of raising them
        assert task.manifest.result != Result.FAILURE, task.manifest.failure_reason
        _await(task.validate())
        assert task.manifest.result == Result.SUCCESS, task.manifest.failure_reason

        return work_path / DESTINATION

    def test_raw_dump(self, dump: Path, tmp_path: Path) -> None:
        out = self._run(tmp_path / 'work', str(dump))
        rows = _count(out, 'count(*), count(DISTINCT id)')
        assert rows == (1000, 1000)

    def test_zip(self, dump: Path, tmp_path: Path) -> None:
        archive = tmp_path / 'demo.zip'
        with zipfile.ZipFile(archive, 'w') as z:
            z.write(dump, 'nested/postgres.dmp')
        out = self._run(tmp_path / 'work', str(archive))
        assert _count(out) == (1000,)

    def test_tar_gz(self, dump: Path, tmp_path: Path) -> None:
        archive = tmp_path / 'demo.tar.gz'
        with tarfile.open(archive, 'w:gz') as t:
            t.add(dump, 'chembl/chembl_postgresql.dmp')
        out = self._run(tmp_path / 'work', str(archive))
        assert _count(out) == (1000,)

    def test_validation_catches_a_short_export(self, dump: Path, tmp_path: Path) -> None:
        """The row count check is the only defence against a partial restore."""
        work = tmp_path / 'work'
        task = self._build(work, str(dump))
        _await(task.run())

        out = work / DESTINATION
        short = out.with_suffix('.short.parquet')
        duckdb.connect().execute(
            'COPY (SELECT * FROM read_parquet($1) LIMIT 10) TO $2 (FORMAT parquet)',
            [str(out), str(short)],
        )
        short.replace(out)

        _await(task.validate())
        assert task.manifest.result == Result.FAILURE
        assert task.manifest.failure_reason is not None
        assert 'has 10 rows, expected 1000' in task.manifest.failure_reason

    def test_an_empty_table_fails_without_allow_empty(self, dump: Path, tmp_path: Path) -> None:
        """A genuinely empty table must not export silently: it might just as well be a missed restore."""
        work = tmp_path / 'work'
        table = TableSpec(table='empty', destination=DESTINATION, columns=['id', 'txt'])
        task = self._build(work, str(dump), tables=[table])

        _await(task.run())

        assert task.manifest.result == Result.FAILURE
        assert task.manifest.failure_reason is not None
        assert 'exported no rows from empty' in task.manifest.failure_reason

    def test_an_empty_table_succeeds_with_allow_empty(self, dump: Path, tmp_path: Path) -> None:
        work = tmp_path / 'work'
        table = TableSpec(table='empty', destination=DESTINATION, columns=['id', 'txt'], allow_empty=True)
        task = self._build(work, str(dump), tables=[table])

        _await(task.run())
        assert task.manifest.result != Result.FAILURE, task.manifest.failure_reason
        _await(task.validate())
        assert task.manifest.result == Result.SUCCESS, task.manifest.failure_reason

        assert _count(work / DESTINATION) == (0,)

    def test_scratch_is_cleaned_up(self, dump: Path, tmp_path: Path) -> None:
        work = tmp_path / 'work'
        self._run(work, str(dump))
        assert not list((work / '.scratch').glob('*'))
