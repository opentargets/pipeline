"""Restore a postgres dump into an ephemeral database and export tables as parquet.

The task spins up a throwaway postgres server (bundled by ``pixeltable-pgserver``,
no docker or system install), restores only the tables it needs from a ``pg_dump``
archive, exports each one to parquet with DuckDB, and deletes the server again.

.. note:: Every large file this task touches lives under
    :py:obj:`otter.config.model.Config.work_path`. On the pipeline VM that is the
    dedicated work disk; the container root filesystem and ``/tmp`` are on a much
    smaller boot disk, so a stray temporary file there will fill it.
"""

import re
import shutil
import subprocess
import tarfile
import time
import zipfile
from contextlib import closing, suppress
from fnmatch import fnmatch
from importlib import resources
from pathlib import Path
from typing import IO, Annotated, Any, Self

import duckdb
from loguru import logger
from otter.manifest.model import Artifact
from otter.storage.synchronous.filesystem import FilesystemStorage
from otter.storage.synchronous.handle import StorageHandle
from otter.task.model import Spec, Task, TaskContext
from otter.task.task_reporter import report
from otter.util.errors import OtterError, TaskAbortedError, TaskValidationError
from otter.validators import file

# the package re-exports these from its __init__ without an __all__, which type
# checkers read as private, so take them from the modules that define them
from pixeltable_pgserver.postgres_server import PostgresServer, get_server
from pixeltable_pgserver.utils import TARGET_POSTGRES_VERSION
from pydantic import BaseModel, Field, field_validator, model_validator

CHUNK_SIZE = 8 * 1024 * 1024
POLL_INTERVAL = 1.0

DUMP_MAGIC = b'PGDMP'
MAX_ARCHIVE_VERSION = (1, 16)
"""Highest ``pg_dump`` archive version the bundled postgres can restore.

``pixeltable-pgserver`` bundles PostgreSQL 18, which emits and reads 1.16. AACT
and ChEMBL currently ship 1.14 and 1.13. A source on a postgres newer than the
bundled one would emit an archive this cannot read, which is what the check is
for.
"""

DATABASE = 'postgres'
"""Database on the ephemeral server the dump is restored into."""

QUERY_PACKAGE = 'pis.sql'
"""Package holding the SQL files named by ``QuerySpec.query``."""

PG_SETTINGS = {
    'fsync': 'off',
    'full_page_writes': 'off',
    'synchronous_commit': 'off',
    'max_wal_size': '8GB',
    'maintenance_work_mem': '2GB',
    'autovacuum': 'off',
    # not a tuning knob: docker gives containers a 64MB /dev/shm, which is not
    # enough for the dynamic shared memory segments postgres allocates for
    # parallel query. duckdb parallelises the scan on its own side anyway.
    'max_parallel_workers_per_gather': '0',
}
"""Settings applied to the ephemeral server once it is up.

Only settings that can be applied with a config reload are listed: the server is
started for us, so there is no hook to change postmaster-level settings without
restarting it.
"""


class PostgresExportError(OtterError):
    """Base class for postgres export errors."""


class TableSpec(BaseModel):
    """One table to restore and export."""

    table: str
    """Name of the table in the source database, without the schema."""
    destination: str
    """Path for the parquet file, relative to the release root."""
    columns: list[str] | None = None
    """Columns to select. If omitted, every column is exported."""
    where: str | None = None
    """Optional SQL predicate, without the ``WHERE`` keyword."""


class QuerySpec(BaseModel):
    """One SQL query to run against the restored database and export."""

    query: str
    """Name of the SQL file in :py:obj:`QUERY_PACKAGE`, without the ``.sql`` suffix."""
    destination: str
    """Path for the parquet file, relative to the release root."""
    requires_tables: Annotated[list[str], Field(min_length=1)]
    """Tables ``pg_restore`` must load for this query to run.

    The restore is selective, so a table a query reads but does not declare will
    simply not be there. ChEMBL's large tables make restoring everything
    impractical, which is why this is explicit rather than inferred.
    """


class PostgresExportSpec(Spec):
    """Configuration fields for the postgres_export task.

    .. note:: ``columns`` and ``where`` are interpolated into the query, not
        bound as parameters, because ``COPY`` takes no bind parameters. The
        configuration file is trusted input written by pipeline developers.
    """

    source: str
    """The dump to restore, relative to the release root. Either a ``pg_dump``
        archive in the custom format, or a zip or tar containing one."""
    archive_member: str = '*.dmp'
    """Name or glob of the dump inside ``source``, when ``source`` is a zip or a
        tar. Ignored when ``source`` is already a dump. Must match exactly one
        member."""
    schema_name: str = 'public'
    """The schema the tables live in."""
    tables: list[TableSpec] = []
    """The tables to restore and export verbatim."""
    queries: list[QuerySpec] = []
    """The queries to run against the restored database and export."""
    jobs: int = 8
    """How many tables ``pg_restore`` loads concurrently."""

    @field_validator('queries')
    @classmethod
    def _sql_files_exist(cls, value: list[QuerySpec]) -> list[QuerySpec]:
        for query in value:
            try:
                _load_query(query.query)
            except PostgresExportError as e:
                # pydantic only turns ValueError/TypeError/AssertionError raised by a
                # validator into a ValidationError; PostgresExportError would otherwise
                # propagate unchanged and skip the spec-level error reporting
                raise ValueError(str(e)) from e
        return value

    @model_validator(mode='after')
    def _has_work(self) -> Self:
        if not self.tables and not self.queries:
            raise ValueError('postgres_export needs at least one of tables or queries')
        return self


def _slug(name: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')


def _sql_str(value: str) -> str:
    """Render a python string as a single quoted SQL literal."""
    escaped = value.replace("'", "''")
    return f"'{escaped}'"


def _load_query(name: str) -> str:
    """Read a SQL file shipped inside the package.

    Reading through ``importlib.resources`` rather than a path relative to this
    module means it works the same whether the package is installed in the image
    or run from the source tree.
    """
    try:
        return resources.files(QUERY_PACKAGE).joinpath(f'{name}.sql').read_text(encoding='utf-8')
    except (FileNotFoundError, ModuleNotFoundError, OSError) as e:
        raise PostgresExportError(f'no SQL file for query {name}: {e}')


def _resolve_archive_member(names: list[str], pattern: str) -> str:
    """Find the single member of an archive matching a name or a glob."""
    matches = [n for n in names if n == pattern or fnmatch(n, pattern)]
    if len(matches) != 1:
        raise PostgresExportError(f'{pattern} matched {len(matches)} members of the archive, expected 1: {names}')
    return matches[0]


def _check_archive_version(dump: Path) -> None:
    """Check the dump is a custom format archive the bundled postgres can read."""
    with dump.open('rb') as f:
        header = f.read(8)

    if len(header) < 8 or header[: len(DUMP_MAGIC)] != DUMP_MAGIC:
        raise PostgresExportError(f'{dump} is not a pg_dump archive')

    version = (header[5], header[6])
    if version > MAX_ARCHIVE_VERSION:
        raise PostgresExportError(
            f'the dump is a version {version[0]}.{version[1]} archive, but the bundled postgres can only read up '
            f'to {MAX_ARCHIVE_VERSION[0]}.{MAX_ARCHIVE_VERSION[1]}; the bundled postgres needs upgrading'
        )
    logger.debug(f'dump is a version {version[0]}.{version[1]} archive')


def _build_restore_args(bin_path: Path, uri: str, dump: Path, tables: list[str], jobs: int) -> list[list[str]]:
    """Build the ``pg_restore`` invocations that load the requested tables.

    The restore runs in two passes. The first restores the schema of the whole
    archive: filtering it by table would skip the ``CREATE SCHEMA`` entry, and any
    type the tables depend on. It is cheap, because indexes and constraints belong
    to the ``post-data`` section, which is never restored: they are the bulk of a
    full restore and nothing here needs them.

    The second pass loads the data of the requested tables only.
    """
    common = [str(bin_path / 'pg_restore'), '--no-owner', '--no-privileges', '--dbname', uri]

    pre_data = [*common, '--section=pre-data', str(dump)]

    data = [*common, '--section=data', '--jobs', str(jobs)]
    for table in tables:
        data += ['--table', table]
    data.append(str(dump))

    return [pre_data, data]


def _build_copy_sql(table: TableSpec, schema_name: str, destination: Path) -> str:
    """Build the DuckDB statement exporting one table to a parquet file."""
    columns = ', '.join(f'"{c}"' for c in table.columns) if table.columns else '*'
    where = f' WHERE {table.where}' if table.where else ''
    query = f'SELECT DISTINCT {columns} FROM pg."{schema_name}"."{table.table}"{where}'  # noqa: S608 trusted config
    return f'COPY ({query}) TO {_sql_str(str(destination))} (FORMAT parquet, COMPRESSION zstd)'


def _scalar(con: duckdb.DuckDBPyConnection, sql: str, parameters: list[Any] | None = None) -> Any:
    """Run a statement that returns a single value, and return it."""
    result = con.execute(sql, parameters).fetchone()
    if result is None:
        raise PostgresExportError(f'no result from: {sql}')
    return result[0]


def _build_count_sql(table: TableSpec, schema_name: str) -> str:
    """Build the DuckDB statement counting the rows of one table in postgres.

    ``postgres_query`` pushes the aggregate down to the server instead of
    transferring the table to count it here.
    """
    inner = f'SELECT count(*) FROM "{schema_name}"."{table.table}"'  # noqa: S608 trusted config
    return f'SELECT * FROM postgres_query({_sql_str("pg")}, {_sql_str(inner)})'  # noqa: S608 trusted config


class PostgresExport(Task):
    """Restore a postgres dump into an ephemeral database and export tables as parquet.

    The ephemeral server exists only for the duration of the task: otter runs each
    task in its own process, so a database cannot be shared with another one. That
    is why a single task exports many tables — the restore is by far the expensive
    part, and doing it once for all of them is the point.

    .. note:: ``destination`` will be prepended with either
        :py:obj:`otter.config.model.Config.release_uri` or
        :py:obj:`otter.config.model.Config.work_path`, like every other task.
    """

    def __init__(
        self,
        spec: PostgresExportSpec,
        context: TaskContext,
    ) -> None:
        super().__init__(spec, context)
        self.spec: PostgresExportSpec
        self.scratch = Path(self.context.config.work_path) / '.scratch' / _slug(self.spec.name)
        self.row_counts: dict[str, int] = {}

    def _check_abort(self) -> None:
        if self.context.abort and self.context.abort.is_set():
            raise TaskAbortedError

    def _copy_stream(self, source: IO[bytes], destination: IO[bytes]) -> None:
        while chunk := source.read(CHUNK_SIZE):
            destination.write(chunk)
            self._check_abort()

    def _copy_to_file(self, source: IO[bytes], destination: Path) -> None:
        with destination.open('wb') as f:
            self._copy_stream(source, f)

    def _stage_archive(self) -> Path:
        """Get the archive onto the local disk."""
        src = StorageHandle(self.spec.source, config=self.context.config)

        # a local archive can be read where it is, which saves copying gigabytes
        # around on local runs
        if isinstance(src.storage, FilesystemStorage):
            logger.info(f'reading archive from {src.absolute}')
            return Path(src.absolute)

        local = self.scratch / 'archive'
        logger.info(f'staging archive from {src.absolute}')
        with src.open('rb') as f:
            self._copy_to_file(f, local)
        return local

    def _extract_dump(self, archive: Path) -> Path:
        """Get the dump out of the archive, if it is in one."""
        with archive.open('rb') as head:
            if head.read(len(DUMP_MAGIC)) == DUMP_MAGIC:
                logger.info('source is a dump, nothing to extract')
                return archive

        dump = self.scratch / 'dump.dmp'

        if zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive) as z:
                member = _resolve_archive_member(z.namelist(), self.spec.archive_member)
                logger.info(f'extracting {member} from zip archive')
                with z.open(member) as source:
                    self._copy_to_file(source, dump)

        elif tarfile.is_tarfile(archive):
            # listing a compressed tar decompresses it, so this reads the archive
            # twice. it is worth it to be able to tell an ambiguous glob from a
            # good one before restoring anything
            with tarfile.open(archive) as t:
                member = _resolve_archive_member(t.getnames(), self.spec.archive_member)
                logger.info(f'extracting {member} from tar archive')
                extracted = t.extractfile(member)
                if extracted is None:
                    raise PostgresExportError(f'{member} is not a regular file')
                self._copy_to_file(extracted, dump)

        else:
            raise PostgresExportError(f'{self.spec.source} is not a dump, a zip or a tar')

        return dump

    def _start_server(self) -> PostgresServer:
        pgdata = self.scratch / 'pgdata'
        logger.info(f'starting an ephemeral postgres {TARGET_POSTGRES_VERSION} server in {pgdata}')
        try:
            return get_server(pgdata, cleanup_mode='delete')
        except Exception as e:
            raise PostgresExportError(f'could not start postgres: {e}')

    def _psql(self, bin_path: Path, uri: str, commands: list[str]) -> None:
        """Run commands through psql, one statement at a time.

        ``ALTER SYSTEM`` cannot run inside a transaction, and psql reading from
        stdin runs each statement on its own.
        """
        subprocess.run(
            # without ON_ERROR_STOP, psql reports success even when a statement failed
            [str(bin_path / 'psql'), '--quiet', '-v', 'ON_ERROR_STOP=1', uri],
            input=';\n'.join(commands).encode(),
            check=True,
            capture_output=True,
        )

    def _tune(self, bin_path: Path, uri: str) -> None:
        logger.debug(f'applying settings to the ephemeral server: {PG_SETTINGS}')
        commands = [f'ALTER SYSTEM SET {k} = {_sql_str(v)}' for k, v in PG_SETTINGS.items()]
        commands.append('SELECT pg_reload_conf()')
        try:
            self._psql(bin_path, uri, commands)
        except subprocess.CalledProcessError as e:
            logger.warning(f'could not apply settings, continuing with the defaults: {e.stderr.decode()}')

    def _run(self, args: list[str], *, strict: bool) -> None:
        """Run a command, letting an abort interrupt it.

        Output goes to a file rather than a pipe, so that a chatty command cannot
        fill a pipe buffer and deadlock while we are polling.
        """
        name = Path(args[0]).name
        log = self.scratch / f'{name}.log'
        logger.debug(f'running {" ".join(args)}')

        with log.open('ab') as f:
            proc = subprocess.Popen(args, stdout=f, stderr=subprocess.STDOUT)
            try:
                while proc.poll() is None:
                    self._check_abort()
                    time.sleep(POLL_INTERVAL)
            except TaskAbortedError:
                proc.terminate()
                raise

        if proc.returncode != 0:
            tail = log.read_text(errors='replace').splitlines()[-20:]
            message = f'{name} exited with {proc.returncode}:\n' + '\n'.join(tail)
            if strict:
                raise PostgresExportError(message)
            logger.warning(message)

    def _restore(self, bin_path: Path, uri: str, dump: Path) -> None:
        tables = [t.table for t in self.spec.tables]
        logger.info(f'restoring {len(tables)} tables into the ephemeral database')

        pre_data, data = _build_restore_args(bin_path, uri, dump, tables, self.spec.jobs)

        # restoring someone else's schema reliably reports errors we do not care
        # about, such as roles that do not exist here, or comments on objects the
        # dump does not contain. the row counts below are what tells us whether
        # the restore actually worked
        self._run(pre_data, strict=False)
        self._run(data, strict=True)

    def _export_table(self, con: duckdb.DuckDBPyConnection, table: TableSpec) -> Artifact:
        source_rows = int(_scalar(con, _build_count_sql(table, self.spec.schema_name)))

        local = Path(StorageHandle(table.destination, config=self.context.config, force_local=True).absolute)
        local.parent.mkdir(parents=True, exist_ok=True)
        rows = int(_scalar(con, _build_copy_sql(table, self.spec.schema_name, local)))

        if source_rows and not rows:
            raise PostgresExportError(
                f'exported no rows from {table.table}, which has {source_rows} in the database: '
                'the restore did not load it'
            )
        logger.info(f'exported {rows} of {source_rows} rows from {table.table} to {table.destination}')
        self.row_counts[table.destination] = rows

        dst = StorageHandle(table.destination, config=self.context.config)
        if dst.absolute != str(local):
            logger.debug(f'uploading {local} to {dst.absolute}')
            # open() streams the upload, unlike write(), which needs the whole
            # object in memory
            with local.open('rb') as f, dst.open('wb') as g:
                self._copy_stream(f, g)

        return Artifact(source=f'{self.spec.source}#{table.table}', destination=dst.absolute)

    def _export(self, uri: str) -> list[Artifact]:
        temp_directory = self.scratch / 'duckdb'
        temp_directory.mkdir(parents=True, exist_ok=True)

        with closing(duckdb.connect(config={'temp_directory': str(temp_directory)})) as con:
            # the extension is baked into the image, so this must not need the
            # network. loading it explicitly makes a missing one fail here rather
            # than silently reaching out for it
            con.execute('LOAD postgres')
            con.execute('SET preserve_insertion_order = false')
            con.execute(f'ATTACH {_sql_str(uri)} AS pg (TYPE postgres, READ_ONLY)')

            artifacts = []
            for table in self.spec.tables:
                self._check_abort()
                artifacts.append(self._export_table(con, table))

        return artifacts

    @report
    def run(self) -> Self:
        self.scratch.mkdir(parents=True, exist_ok=True)
        dump = self._extract_dump(self._stage_archive())
        _check_archive_version(dump)

        server = None
        try:
            server = self._start_server()
            uri = server.get_uri(database=DATABASE)
            self._tune(server.bin_path, uri)
            self._restore(server.bin_path, uri, dump)
            self.artifacts = self._export(uri)
        finally:
            if server is not None:
                # never let a failure to clean up hide the reason we are here
                with suppress(Exception):
                    server.cleanup()
            shutil.rmtree(self.scratch, ignore_errors=True)

        return self

    @report
    def validate(self) -> Self:
        """Check every table landed in the release with the number of rows it was exported with.

        The ephemeral database is long gone by the time validation runs, so the
        counts come from the row counts recorded during the run.
        """
        with closing(duckdb.connect()) as con:
            for table in self.spec.tables:
                file.exists(table.destination, config=self.context.config)

                h = StorageHandle(table.destination, config=self.context.config)
                if not h.stat().size:
                    raise TaskValidationError(f'{table.destination} is empty')

                local = StorageHandle(table.destination, config=self.context.config, force_local=True).absolute
                expected = self.row_counts.get(table.destination)
                if expected is None or not Path(local).is_file():
                    logger.warning(f'no local copy of {table.destination}, cannot check its row count')
                    continue

                rows = int(_scalar(con, 'SELECT count(*) FROM read_parquet(?)', [local]))
                if rows != expected:
                    raise TaskValidationError(f'{table.destination} has {rows} rows, expected {expected}')

        return self
