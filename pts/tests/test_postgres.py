"""Tests for the ephemeral postgres helper.

Deliberately small. Each test here pins something that, if it broke, would
either change the data a step reads or leave a server and tens of gigabytes
behind. Argument shapes and happy paths that the round trip already exercises
are not repeated.
"""

import subprocess
import tarfile
import zipfile
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from otter.config.model import Config
from pixeltable_pgserver.postgres_server import get_server
from pytest_mock import MockerFixture

from pts.postgres import (
    DUMP_MAGIC,
    MAX_ARCHIVE_VERSION,
    PostgresError,
    _build_restore_args,
    _build_select_sql,
    _check_archive_version,
    _check_order_by,
    _resolve_archive_member,
    read_dump_tables,
    restored_dump,
)

ROWS = 1000
"""Rows in the demo table the round trip builds."""

DISTINCT_TXT = 7
"""Distinct values of ``txt`` across those rows."""


def test_reads_are_always_distinct() -> None:
    """Every row count downstream of here was measured with duplicates removed."""
    assert _build_select_sql('studies', 'ctgov', ['nct_id', 'phase']) == (
        'SELECT DISTINCT "nct_id", "phase" FROM "ctgov"."studies"'
    )


def test_an_ordered_read_says_so_in_the_sql() -> None:
    """Without ORDER BY the row order is the plan's business, and it reaches published arrays."""
    assert _build_select_sql('warning_refs', 'public', ['warnref_id', 'ref_id'], ['warnref_id']) == (
        'SELECT DISTINCT "warnref_id", "ref_id" FROM "public"."warning_refs" ORDER BY "warnref_id"'
    )


def test_ordering_by_a_column_outside_the_projection_is_caught_before_the_restore() -> None:
    """A SELECT DISTINCT can only order by what it selects, and a restore takes minutes."""
    with pytest.raises(PostgresError, match='not in its projection'):
        _check_order_by({'warning_refs': ['warning_id']}, {'warning_refs': ['warnref_id']})


def test_ordering_a_table_that_is_not_being_read_is_caught() -> None:
    """Almost always a typo, and silently ignoring it would leave the read unordered."""
    with pytest.raises(PostgresError, match='not one of the tables being read'):
        _check_order_by({'warning_refs': None}, {'warning_ref': ['warnref_id']})


def test_restores_only_the_requested_tables_and_skips_the_indexes() -> None:
    """The two-pass restore is the reason this is minutes rather than hours.

    Pass one takes the whole schema, because filtering it would drop the
    ``CREATE SCHEMA`` entry and any type the tables need. Pass two takes the data
    of the requested tables only. Neither touches ``post-data``, which is where
    indexes and constraints live.
    """
    pre_data, data = _build_restore_args(
        Path('/bin'), 'postgresql://x', Path('/d.dmp'), ['studies', 'designs'], 'ctgov', 8
    )

    assert '--section=pre-data' in pre_data
    assert '--table' not in pre_data

    assert '--section=data' in data
    assert [data[i + 1] for i, arg in enumerate(data) if arg == '--table'] == ['studies', 'designs']

    for p in (pre_data, data):
        assert not any('post-data' in arg for arg in p)
        # someone else's dump names owners and roles that do not exist here
        assert '--no-owner' in p
        assert '--no-privileges' in p


def test_an_ambiguous_archive_member_is_an_error() -> None:
    """Picking one of several dumps silently would restore the wrong data."""
    with pytest.raises(PostgresError, match='matched 2 members'):
        _resolve_archive_member(['a.dmp', 'b.dmp'], '*.dmp')


@pytest.mark.parametrize(
    ('minor', 'readable'),
    [(13, True), (14, True), (MAX_ARCHIVE_VERSION[1], True), (MAX_ARCHIVE_VERSION[1] + 1, False)],
    ids=['chembl', 'aact', 'bundled', 'too_new'],
)
def test_the_archive_version_guard(minor: int, readable: bool, tmp_path: Path) -> None:
    """A source on a newer postgres must say so, not fail somewhere in the restore."""
    dump = tmp_path / 'd.dmp'
    dump.write_bytes(DUMP_MAGIC + bytes([1, minor, 0]) + b'\x00' * 32)

    if readable:
        _check_archive_version(dump)
    else:
        with pytest.raises(PostgresError, match='bundled postgres needs upgrading'):
            _check_archive_version(dump)


def _as_zip(dump: Path, tmp_path: Path) -> Path:
    archive = tmp_path / 'demo.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        z.write(dump, 'nested/postgres.dmp')
    return archive


def _as_tar_gz(dump: Path, tmp_path: Path) -> Path:
    archive = tmp_path / 'demo.tar.gz'
    with tarfile.open(archive, 'w:gz') as t:
        t.add(dump, 'chembl/chembl_postgresql.dmp')
    return archive


@pytest.mark.pgserver
class TestRoundTrip:
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
            rows = f"SELECT i, 'x' || (i % {DISTINCT_TXT}) FROM generate_series(1, {ROWS}) i"  # noqa: S608 fixture
            server.psql(f'CREATE SCHEMA demo;CREATE TABLE demo.t (id int, txt text);INSERT INTO demo.t {rows};')
            dump = path / 'demo.dmp'
            subprocess.run([str(server.bin_path / 'pg_dump'), '-Fc', '-f', str(dump), server.get_uri()], check=True)
            yield dump
        finally:
            server.cleanup()

    def _read(self, source: Path, scratch_root: Path, columns: list[str]) -> pl.DataFrame:
        return read_dump_tables(str(source), {'t': columns}, schema_name='demo', scratch_root=scratch_root)['t']

    def test_reads_a_restored_table_with_polars(self, dump: Path, tmp_path: Path) -> None:
        df = self._read(dump, tmp_path, ['id', 'txt'])
        assert df.height == ROWS
        assert df.columns == ['id', 'txt']
        # postgres `integer` must come back as Int32, not widened to Int64: steps such as
        # drug_warning rely on this narrowing to reach the released `year` column untouched.
        assert df.schema['id'] == pl.Int32

    def test_an_ordered_read_comes_back_ordered(self, dump: Path, tmp_path: Path) -> None:
        """The rows arrive in the asked-for order, not the plan's."""
        df = read_dump_tables(
            str(dump),
            {'t': ['id', 'txt']},
            schema_name='demo',
            order_by={'t': ['id']},
            scratch_root=tmp_path,
        )['t']
        assert df.get_column('id').to_list() == sorted(df.get_column('id').to_list())

    def test_reads_are_distinct_end_to_end(self, dump: Path, tmp_path: Path) -> None:
        """Projecting to ``txt`` alone collapses the rows, as it does for the real sources."""
        assert self._read(dump, tmp_path, ['txt']).height == DISTINCT_TXT

    @pytest.mark.parametrize('pack', [_as_zip, _as_tar_gz], ids=['aact_ships_a_zip', 'chembl_ships_a_tar_gz'])
    def test_reads_a_dump_wrapped_in_an_archive(
        self, pack: Callable[[Path, Path], Path], dump: Path, tmp_path: Path
    ) -> None:
        assert self._read(pack(dump, tmp_path), tmp_path, ['id', 'txt']).height == ROWS

    def test_an_empty_table_is_an_error(self, tmp_path: Path) -> None:
        """A restore that loads nothing is silent: pg_restore is happy and the query works."""
        source = tmp_path / 'source'
        source.mkdir()
        server = get_server(source / 'pgdata', cleanup_mode='delete')
        try:
            server.psql('CREATE SCHEMA demo; CREATE TABLE demo.empty (id int);')
            dump = tmp_path / 'empty.dmp'
            subprocess.run([str(server.bin_path / 'pg_dump'), '-Fc', '-f', str(dump), server.get_uri()], check=True)
        finally:
            server.cleanup()

        with pytest.raises(PostgresError, match='came back empty'):
            read_dump_tables(str(dump), {'empty': ['id']}, schema_name='demo', scratch_root=tmp_path)

    def test_everything_is_cleaned_up_when_the_body_raises(self, dump: Path, tmp_path: Path) -> None:
        """Otherwise a failed read leaves a postgres running and the dump on disk."""
        scratch_root = tmp_path / 'scratch'
        scratch_root.mkdir()

        with (
            pytest.raises(ZeroDivisionError),
            restored_dump(str(dump), tables=['t'], schema_name='demo', scratch_root=scratch_root),
        ):
            1 / 0  # noqa: B018 the point is to leave the block by raising

        assert not list(scratch_root.iterdir())


class TestRestoreArgsImprovements:
    def test_the_data_pass_uses_strict_names(self) -> None:
        """A --table matching nothing exits 0 and restores silently nothing."""
        _, data = _build_restore_args(Path('/bin'), 'postgresql://x', Path('d.dmp'), ['studies'], 'ctgov', 8)
        assert '--strict-names' in data

    def test_the_data_pass_is_schema_qualified(self) -> None:
        """pg_restore's --table matches the bare name across every schema in the archive."""
        _, data = _build_restore_args(Path('/bin'), 'postgresql://x', Path('d.dmp'), ['studies'], 'ctgov', 8)
        assert '--schema' in data
        assert data[data.index('--schema') + 1] == 'ctgov'

    def test_the_pre_data_pass_is_not_schema_qualified(self) -> None:
        """Filtering pre-data would skip CREATE SCHEMA and any types the tables need."""
        pre_data, _ = _build_restore_args(Path('/bin'), 'postgresql://x', Path('d.dmp'), ['studies'], 'ctgov', 8)
        assert '--schema' not in pre_data


class _CapturedError(Exception):
    """Raised by the stub once it has the kwargs, to stop the transformer there."""


WORK_PATH = Path('/mnt/disks/work')
"""What `work_path` is on the pipeline VM, where the work disk is mounted."""

CALL_SITES = {
    'chembl_target_class_dump': ('pts.transformers.chembl_target_class_dump', Path('chembl.tar.gz'), {}),
    'drug_warning': ('pts.transformers.drug_warning', Path('chembl.tar.gz'), Path('out.parquet')),
    'drug_mechanism_of_action': (
        'pts.transformers.drug_mechanism_of_action',
        {'chembl': Path('chembl.tar.gz'), 'target': Path('genes.parquet')},
        Path('out.parquet'),
    ),
    'chembl_molecule': (
        'pts.transformers.chembl_molecule',
        {'chembl': Path('chembl.tar.gz'), 'drugbank': Path('drugbank.csv.gz')},
        Path('out.parquet'),
    ),
}
"""The four transformers that restore a dump, with the arguments they take."""


@pytest.mark.parametrize('transformer', list(CALL_SITES), ids=list(CALL_SITES))
def test_the_restore_scratch_goes_on_the_work_disk(transformer: str, mocker: MockerFixture) -> None:
    """A restore that defaults to /tmp fills the VM's boot disk and dies mid-restore.

    On the pipeline VM the large work disk is mounted at `work_path`, while the
    container root filesystem -- `/tmp` included -- is on a boot disk an order of
    magnitude smaller. The archive, the extracted dump and the whole pgdata
    directory all land in the scratch, so every one of these transformers has to
    point it at `work_path` rather than take `tempfile`'s default.
    """
    module_name, source, destination = CALL_SITES[transformer]
    module = __import__(module_name, fromlist=['read_dump_tables'])

    captured: dict[str, Any] = {}

    def stub(*args: Any, **kwargs: Any) -> None:
        captured.update(kwargs)
        raise _CapturedError

    mocker.patch.object(module, 'read_dump_tables', stub)
    config = Config(step=transformer, steps=[transformer], work_path=WORK_PATH)

    with pytest.raises(_CapturedError):
        getattr(module, transformer)(source, destination, {}, config)

    assert captured.get('scratch_root') == WORK_PATH
