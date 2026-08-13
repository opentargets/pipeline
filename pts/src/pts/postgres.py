"""Read tables straight out of a ``pg_dump`` archive.

This module gives a step one. :func:`read_dump_tables` is the whole job in a
single call — restore these tables, hand me the dataframes — and is what a step
usually wants. :func:`restored_dump` underneath it spins up a throwaway server
(bundled by ``pixeltable-pgserver``), restores only the tables it was asked for,
yields a connection URI, and deletes the server again.

.. note:: The archive, the dump and the database are all large, and all of them
    go in a temporary directory that is removed again on the way out. A caller
    running on the pipeline VM **must** pass ``scratch_root=config.work_path``:
    that is the dedicated work disk, while the container root filesystem and
    ``/tmp``, where :py:mod:`tempfile` would otherwise put this, are on a much
    smaller boot disk that a single restore would fill. ``TMPDIR`` is the same
    knob from outside the process. A full ChEMBL restore wants tens of gigabytes.
"""

from __future__ import annotations

import shutil
import subprocess
import tarfile
import tempfile
import zipfile
from contextlib import contextmanager, suppress
from fnmatch import fnmatch
from pathlib import Path
from typing import IO, TYPE_CHECKING
from urllib.parse import quote

import polars as pl
from loguru import logger
from otter.storage.synchronous.filesystem import FilesystemStorage
from otter.storage.synchronous.handle import StorageHandle
from otter.util.errors import OtterError

# the package re-exports these from its __init__ without an __all__, which type
# checkers read as private, so take them from the modules that define them
from pixeltable_pgserver.postgres_server import PostgresServer, get_server
from pixeltable_pgserver.utils import TARGET_POSTGRES_VERSION

if TYPE_CHECKING:
    from collections.abc import Generator, Mapping, Sequence

CHUNK_SIZE = 8 * 1024 * 1024

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

PG_SETTINGS = {
    'fsync': 'off',
    'full_page_writes': 'off',
    'synchronous_commit': 'off',
    'max_wal_size': '8GB',
    'maintenance_work_mem': '2GB',
    'autovacuum': 'off',
    # not a tuning knob: docker gives containers a 64MB /dev/shm, which is not
    # enough for the dynamic shared memory segments postgres allocates for
    # parallel query, and nothing here needs it
    'max_parallel_workers_per_gather': '0',
}
"""Settings applied to the ephemeral server once it is up.

Only settings that can be applied with a config reload are listed: the server is
started for us, so there is no hook to change postmaster-level settings without
restarting it.
"""

LOG_TAIL = 20
"""How many lines of a failed command's output to put in the error."""


class PostgresError(OtterError):
    """Base class for postgres errors."""


def _sql_str(value: str) -> str:
    """Render a python string as a single quoted SQL literal."""
    escaped = value.replace("'", "''")
    return f"'{escaped}'"


def _resolve_archive_member(names: list[str], pattern: str) -> str:
    """Find the single member of an archive matching a name or a glob."""
    matches = [n for n in names if n == pattern or fnmatch(n, pattern)]
    if len(matches) != 1:
        raise PostgresError(f'{pattern} matched {len(matches)} members of the archive, expected 1: {names}')
    return matches[0]


def _check_archive_version(dump: Path) -> None:
    """Check the dump is a custom format archive the bundled postgres can read."""
    with dump.open('rb') as f:
        header = f.read(8)

    if len(header) < 8 or header[: len(DUMP_MAGIC)] != DUMP_MAGIC:
        raise PostgresError(f'{dump} is not a pg_dump archive')

    version = (header[5], header[6])
    if version > MAX_ARCHIVE_VERSION:
        raise PostgresError(
            f'the dump is a version {version[0]}.{version[1]} archive, but the bundled postgres can only read up '
            f'to {MAX_ARCHIVE_VERSION[0]}.{MAX_ARCHIVE_VERSION[1]}; the bundled postgres needs upgrading'
        )
    logger.debug(f'dump is a version {version[0]}.{version[1]} archive')


def _build_restore_args(
    bin_path: Path, uri: str, dump: Path, tables: Sequence[str], schema_name: str, jobs: int
) -> list[list[str]]:
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

    # --strict-names: without it, a --table matching nothing in the archive exits
    # 0 and restores silently nothing.
    # --schema on the data pass ONLY: pg_restore's --table matches the bare name
    # across every schema in the archive, so an unqualified restore loads every
    # same-named table from every other schema and never reads them. The pre-data
    # pass stays unqualified on purpose -- see the docstring above.
    data = [*common, '--section=data', '--strict-names', '--schema', schema_name, '--jobs', str(jobs)]
    for table in tables:
        data += ['--table', table]
    data.append(str(dump))

    return [pre_data, data]


def _copy_stream(source: IO[bytes], destination: IO[bytes]) -> None:
    while chunk := source.read(CHUNK_SIZE):
        destination.write(chunk)


def _copy_to_file(source: IO[bytes], destination: Path) -> None:
    with destination.open('wb') as f:
        _copy_stream(source, f)


def _stage_archive(source: str, scratch: Path) -> Path:
    """Get the archive onto the local disk."""
    src = StorageHandle(source)

    # a local archive can be read where it is, which saves copying gigabytes
    # around on local runs
    if isinstance(src.storage, FilesystemStorage):
        logger.info(f'reading archive from {src.absolute}')
        return Path(src.absolute)

    local = scratch / 'archive'
    logger.info(f'staging archive from {src.absolute}')
    with src.open('rb') as f:
        _copy_to_file(f, local)
    return local


def _extract_dump(archive: Path, scratch: Path, member_pattern: str) -> Path:
    """Get the dump out of the archive, if it is in one."""
    with archive.open('rb') as head:
        if head.read(len(DUMP_MAGIC)) == DUMP_MAGIC:
            logger.info('source is a dump, nothing to extract')
            return archive

    dump = scratch / 'dump.dmp'

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            member = _resolve_archive_member(z.namelist(), member_pattern)
            logger.info(f'extracting {member} from zip archive')
            with z.open(member) as source:
                _copy_to_file(source, dump)

    elif tarfile.is_tarfile(archive):
        # listing a compressed tar decompresses it, so this reads the archive
        # twice. it is worth it to be able to tell an ambiguous glob from a
        # good one before restoring anything
        with tarfile.open(archive) as t:
            member = _resolve_archive_member(t.getnames(), member_pattern)
            logger.info(f'extracting {member} from tar archive')
            extracted = t.extractfile(member)
            if extracted is None:
                raise PostgresError(f'{member} is not a regular file')
            _copy_to_file(extracted, dump)

    else:
        raise PostgresError(f'{archive} is not a dump, a zip or a tar')

    return dump


def _start_server(scratch: Path) -> PostgresServer:
    pgdata = scratch / 'pgdata'
    logger.info(f'starting an ephemeral postgres {TARGET_POSTGRES_VERSION} server in {pgdata}')
    try:
        return get_server(pgdata, cleanup_mode='delete')
    except Exception as e:
        raise PostgresError(f'could not start postgres: {e}') from e


def _client_uri(server: PostgresServer) -> str:
    """Render the server's connection URI in a form polars can parse.

    ``pixeltable-pgserver`` starts postgres listening on a unix socket and
    nothing else, and spells the URI the way libpq documents that: an empty host,
    with the socket directory in a ``host`` parameter::

        postgresql://postgres:@/postgres?host=/socket/dir

    connectorx, which backs :py:func:`polars.read_database_uri`, runs that
    through a URL parser that rejects an empty host outright — ``parse error:
    empty host``. libpq's other spelling for the same connection, the
    percent-encoded socket directory in the host position, is understood by both,
    so hand that one out instead.

    ``pg_restore`` and ``psql`` are given the server's own URI, not this one:
    they are libpq, so either spelling works, and there is no reason to reformat
    what the server told us.
    """
    info = server.get_postmaster_info()
    if info.socket_dir is None:
        # windows, where the server listens on a port and the plain uri is fine
        return server.get_uri(database=DATABASE)
    return f'postgresql://{server.postgres_user}@{quote(str(info.socket_dir), safe="")}:{info.port}/{DATABASE}'


def _psql(bin_path: Path, uri: str, commands: list[str]) -> None:
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


def _tune(bin_path: Path, uri: str) -> None:
    logger.debug(f'applying settings to the ephemeral server: {PG_SETTINGS}')
    commands = [f'ALTER SYSTEM SET {k} = {_sql_str(v)}' for k, v in PG_SETTINGS.items()]
    commands.append('SELECT pg_reload_conf()')
    try:
        _psql(bin_path, uri, commands)
    except subprocess.CalledProcessError as e:
        logger.warning(f'could not apply settings, continuing with the defaults: {e.stderr.decode()}')


def _run(args: list[str], scratch: Path, *, strict: bool) -> None:
    """Run a command, sending its output to a file.

    Output goes to a file rather than a pipe, so that a chatty command cannot
    fill a pipe buffer and stall.
    """
    name = Path(args[0]).name
    log = scratch / f'{name}.log'
    logger.debug(f'running {" ".join(args)}')

    with log.open('ab') as f:
        returncode = subprocess.run(args, stdout=f, stderr=subprocess.STDOUT, check=False).returncode

    if returncode != 0:
        tail = log.read_text(errors='replace').splitlines()[-LOG_TAIL:]
        message = f'{name} exited with {returncode}:\n' + '\n'.join(tail)
        if strict:
            raise PostgresError(message)
        logger.warning(message)


def _restore(
    bin_path: Path, uri: str, dump: Path, tables: Sequence[str], schema_name: str, scratch: Path, jobs: int
) -> None:
    logger.info(f'restoring {len(tables)} tables into the ephemeral database')

    pre_data, data = _build_restore_args(bin_path, uri, dump, tables, schema_name, jobs)

    # restoring someone else's schema reliably reports errors we do not care
    # about, such as roles that do not exist here, or comments on objects the
    # dump does not contain. whether the tables came back with any rows in them
    # is what tells us the restore actually worked, and that is the caller's
    # check to make
    _run(pre_data, scratch, strict=False)
    _run(data, scratch, strict=True)


@contextmanager
def restored_dump(
    source: str,
    tables: Sequence[str],
    *,
    schema_name: str,
    archive_member: str = '*.dmp',
    jobs: int = 8,
    scratch_root: str | Path | None = None,
) -> Generator[str]:
    """Restore ``tables`` from a ``pg_dump`` archive and yield a connection URI.

    The server, and every temporary file behind it, are gone by the time this
    returns, so read everything you need before leaving the block.

    Args:
        source: The dump to restore: a ``pg_dump`` archive in the custom format,
            or a zip or a tar containing one. Read through a
            :py:class:`otter.storage.synchronous.handle.StorageHandle`, so it can
            live in a bucket; a local one is read where it is rather than copied.
        tables: Names of the tables to restore, without their schema. Only these
            are loaded — a full restore of AACT or ChEMBL would take far longer
            than any step that needs a handful of their tables can justify.
        schema_name: The schema the tables live in.
        archive_member: Name or glob of the dump inside ``source``, when
            ``source`` is a zip or a tar. Must match exactly one member.
        jobs: How many tables ``pg_restore`` loads concurrently.
        scratch_root: Where to create the temporary directory. Defaults to
            wherever :py:mod:`tempfile` puts things, which is what ``TMPDIR``
            controls. Point it somewhere with room if the default is too small.

    Yields:
        A connection URI for the restored database, in a spelling
        :py:func:`polars.read_database_uri` accepts.
    """
    scratch = Path(tempfile.mkdtemp(prefix='pts-postgres-', dir=scratch_root))
    server = None
    try:
        dump = _extract_dump(_stage_archive(source, scratch), scratch, archive_member)
        _check_archive_version(dump)

        server = _start_server(scratch)
        uri = server.get_uri(database=DATABASE)
        _tune(server.bin_path, uri)
        _restore(server.bin_path, uri, dump, tables, schema_name, scratch, jobs)

        yield _client_uri(server)
    finally:
        if server is not None:
            # never let a failure to clean up hide the reason we are here
            with suppress(Exception):
                server.cleanup()
        shutil.rmtree(scratch, ignore_errors=True)


def _build_select_sql(
    table: str, schema_name: str, columns: Sequence[str] | None = None, order_by: Sequence[str] | None = None
) -> str:
    """Build the statement that reads one table.

    ``DISTINCT`` is not an optimisation. These dumps carry rows that become
    duplicates once the columns a step cares about are projected out of them, and
    every row count downstream of here was measured with those removed. It is
    part of the contract, which is why it is spelled out here and pinned by a
    test rather than left to whatever a query helper does by default.

    ``ORDER BY`` is not one either. Without it the row order is whatever the plan
    happens to produce -- a hash aggregate over the physical order of a restored
    dump -- and a caller that collects a column into a list hands that
    nondeterminism straight to a published artefact. See ``order_by`` in
    :func:`read_dump_tables` for when to ask for it.
    """
    selected = ', '.join(f'"{c}"' for c in columns) if columns else '*'
    sql = f'SELECT DISTINCT {selected} FROM "{schema_name}"."{table}"'  # noqa: S608 trusted caller
    if order_by:
        sql += ' ORDER BY ' + ', '.join(f'"{c}"' for c in order_by)
    return sql


def _check_order_by(tables: Mapping[str, Sequence[str] | None], order_by: Mapping[str, Sequence[str]]) -> None:
    """Reject an ordering that names a table or a column that is not being read.

    ``SELECT DISTINCT`` can only order by expressions in the select list, so a
    column outside the projection is a postgres error in the middle of the read.
    Catching it here says which table and column, before anything is restored.
    """
    for table, columns in order_by.items():
        if table not in tables:
            raise PostgresError(f'order_by names {table}, which is not one of the tables being read: {list(tables)}')
        projection = tables[table]
        if projection is None:
            continue
        if missing := [c for c in columns if c not in projection]:
            raise PostgresError(
                f'order_by names {missing} for {table}, which is not in its projection: {list(projection)}. '
                f'A SELECT DISTINCT can only be ordered by columns it selects.'
            )


def read_dump_tables(
    source: str,
    tables: Mapping[str, Sequence[str] | None],
    *,
    schema_name: str,
    archive_member: str = '*.dmp',
    order_by: Mapping[str, Sequence[str]] | None = None,
    scratch_root: str | Path | None = None,
) -> dict[str, pl.DataFrame]:
    """Restore tables from a ``pg_dump`` archive and read each one into polars.

    Args:
        source: The dump to restore. See :func:`restored_dump`.
        tables: Table name to the columns to read from it, or ``None`` for all of
            them. Only these tables are restored.
        schema_name: The schema the tables live in.
        archive_member: Name or glob of the dump inside ``source``, when
            ``source`` is a zip or a tar.
        order_by: Table name to the columns to order that table's read by. Rows
            come back in whatever order the plan produced otherwise, which is a
            function of the postgres version and of the physical order of the
            dump, and is not stable across releases of either.

            **Every table whose row order can reach the output needs an entry
            here** — anything a caller aggregates into a list without sorting it
            afterwards. Pick columns that are unique together, usually the
            table's own key, so the ordering is total and the read is
            reproducible. It is opt-in rather than the default because ordering
            a large projection (``compound_structures`` is several gigabytes)
            costs a sort that most reads have no use for.
        scratch_root: See :func:`restored_dump`.

    Returns:
        Table name to its contents.

    Raises:
        PostgresError: If a table came back empty. A restore that loads nothing
            is otherwise silent — ``pg_restore`` is happy, the query works, and
            the step carries on with no rows — so it is checked here. A caller
            that genuinely expects an empty table wants :func:`restored_dump`.
            Also if ``order_by`` names a table or a column that is not being read.
    """
    order_by = order_by or {}
    _check_order_by(tables, order_by)

    with restored_dump(
        source, tables=list(tables), schema_name=schema_name, archive_member=archive_member, scratch_root=scratch_root
    ) as uri:
        frames = {
            name: pl.read_database_uri(_build_select_sql(name, schema_name, columns, order_by.get(name)), uri)
            for name, columns in tables.items()
        }

    for name, df in frames.items():
        logger.info(f'read {df.height} rows from {schema_name}.{name}')
        if df.is_empty():
            raise PostgresError(f'{schema_name}.{name} came back empty: the restore did not load it')

    return frames
