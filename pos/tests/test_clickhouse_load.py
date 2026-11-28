"""clickhouse_load creates a served view's landing table from its model, loads it and builds the view's table."""

from typing import Annotated
from unittest.mock import MagicMock

import polars as pl
import pytest
from clickhouse_connect.driver.exceptions import DatabaseError, OperationalError
from loguru import logger
from ot_datamodels import ClickHouseDictionary, Serving
from ot_datamodels.base import DatasetModel, PrimaryKey

from pos import clickhouse_ddl
from pos.clickhouse_ddl import uniqueness_sql
from pos.tasks import clickhouse_load
from pos.tasks.clickhouse_load import ClickhouseLoad, ClickhouseLoadError, ClickhouseLoadSpec


class Toy(DatasetModel):
    dataset_name = 'toy_view'
    dataset_title = 'Toy'
    dataset_description = 'Toy.'
    example = {}
    clickhouse = Serving(engine='mergetree', order_by=('id',))

    id: Annotated[str, PrimaryKey()]


class Declared(DatasetModel):
    dataset_name = 'declared_view'
    dataset_title = 'Declared'
    dataset_description = 'Declared.'
    example = {}
    clickhouse = Serving(
        engine='rocksdb', dictionaries=(ClickHouseDictionary('declared_names', key='id', attributes=('name',)),)
    )

    id: Annotated[str, PrimaryKey()]
    name: str
    target: str | None


class ByTarget(DatasetModel):
    dataset_name = 'by_target_view'
    dataset_title = 'By target'
    dataset_description = 'By target.'
    example = {}
    clickhouse = Serving(engine='mergetree', order_by=('target', 'id'))

    id: str
    name: str
    target: str | None


class PppOnly(DatasetModel):
    dataset_name = 'ppp_only_view'
    dataset_title = 'PPP only'
    dataset_description = 'PPP only.'
    example = {}
    products = ('ppp',)
    clickhouse = Serving(engine='mergetree', order_by=('id',))

    id: str


MODELS = {m.dataset_name: m for m in (Toy, Declared, ByTarget, PppOnly)}
TOY_CREATE = 'CREATE TABLE IF NOT EXISTS `toy_view_log` (`id` String) ENGINE = Log'


@pytest.fixture(autouse=True)
def models(monkeypatch):
    monkeypatch.setattr(clickhouse_ddl, 'discover_models', lambda: MODELS)
    monkeypatch.setattr(clickhouse_ddl, 'discover_datasets', dict)


def _spec(dataset: str = 'toy_view', product: str = 'platform', **kw) -> ClickhouseLoadSpec:
    return ClickhouseLoadSpec(
        name=f'clickhouse_load {dataset}', dataset=dataset, data_dir_parent='release_data', product=product, **kw
    )


def _task(tmp_path, spec: ClickhouseLoadSpec | None = None) -> ClickhouseLoad:
    return ClickhouseLoad(spec or _spec(), MagicMock(config=MagicMock(work_path=tmp_path)))


def _run(task):
    ClickhouseLoad.run.__wrapped__(task)  # ty: ignore[unresolved-attribute]


def _parts(tmp_path, dataset: str, n: int) -> None:
    data = tmp_path / 'release_data' / 'view' / dataset
    data.mkdir(parents=True)
    for i in range(n):
        (data / f'part-{i}.parquet').write_bytes(b'')


@pytest.fixture
def info():
    lines: list[str] = []
    sink = logger.add(lambda m: lines.append(m.record['message']), level='INFO')
    yield lines
    logger.remove(sink)


def test_clickhouse_load_fails_on_no_parquet(tmp_path, monkeypatch):
    (tmp_path / 'release_data' / 'view' / 'toy_view').mkdir(parents=True)
    client = MagicMock()
    monkeypatch.setattr(clickhouse_load, 'connect', lambda **kw: client)
    with pytest.raises(ClickhouseLoadError, match=r'dataset toy_view in .*release_data/view/toy_view'):
        _run(_task(tmp_path))
    client.command.assert_not_called()


def test_an_unknown_view_fails_at_spec_time(tmp_path):
    with pytest.raises(ClickhouseLoadError, match='unable to load config for evidence'):
        _task(tmp_path, _spec('evidence'))


def test_a_view_not_served_for_the_product_fails_at_spec_time(tmp_path):
    with pytest.raises(ClickhouseLoadError, match='ppp_only_view: is not served for platform'):
        _task(tmp_path, _spec('ppp_only_view'))
    _task(tmp_path, _spec('ppp_only_view', 'ppp'))


def _recorded_task(tmp_path, monkeypatch, failures):
    """A load of two files whose inserts raise the given errors in turn, then succeed."""
    _parts(tmp_path, 'toy_view', 2)
    client = MagicMock()
    monkeypatch.setattr(clickhouse_load, 'connect', lambda **kw: client)
    events = []
    errors = list(failures)

    def insert(c, table, path, fmt, settings):
        name = path.rsplit('/', 1)[1]
        if errors and name == 'part-1.parquet':
            events.append(('failed', name))
            raise errors.pop(0)
        events.append(('insert', table, name))

    monkeypatch.setattr(clickhouse_load, 'insert_file', insert)
    client.command.side_effect = lambda sql: events.append(('command', sql))
    return _task(tmp_path), events


def test_the_landing_table_is_created_loaded_then_built_into_the_view_s_table(tmp_path, monkeypatch, info):
    task, events = _recorded_task(tmp_path, monkeypatch, [])
    _run(task)
    assert events == [
        ('command', 'DROP TABLE IF EXISTS `toy_view_log` SYNC'),
        ('command', TOY_CREATE),
        ('insert', 'toy_view_log', 'part-0.parquet'),
        ('insert', 'toy_view_log', 'part-1.parquet'),
        ('command', 'SELECT count() FROM `toy_view_log`'),
        ('command', 'DROP TABLE IF EXISTS `toy_view` SYNC'),
        ('command', 'CREATE TABLE `toy_view` (`id` String) ENGINE = MergeTree() ORDER BY (`id`)'),
        ('command', 'INSERT INTO `toy_view` SELECT `id` FROM `toy_view_log`'),
        ('command', 'OPTIMIZE TABLE `toy_view` FINAL'),
        ('command', 'DROP TABLE `toy_view_log` SYNC'),
        ('command', 'SELECT count() FROM `toy_view`'),
    ]
    assert TOY_CREATE in info


def test_clickhouse_load_truncates_and_reloads_every_file_after_a_failed_insert(tmp_path, monkeypatch):
    task, events = _recorded_task(tmp_path, monkeypatch, [OperationalError('timed out')])
    _run(task)
    truncate = events.index(('command', 'TRUNCATE TABLE `toy_view_log`'))
    assert ('failed', 'part-1.parquet') in events[:truncate]
    assert events[truncate + 1 : truncate + 4] == [
        ('insert', 'toy_view_log', 'part-0.parquet'),
        ('insert', 'toy_view_log', 'part-1.parquet'),
        ('command', 'SELECT count() FROM `toy_view_log`'),
    ]


def test_clickhouse_load_fails_when_the_reload_fails_too(tmp_path, monkeypatch):
    task, events = _recorded_task(tmp_path, monkeypatch, [DatabaseError('code: 159'), DatabaseError('code: 159')])
    with pytest.raises(ClickhouseLoadError) as raised:
        _run(task)
    assert [e for e in events if e[0] == 'command'] == [
        ('command', 'DROP TABLE IF EXISTS `toy_view_log` SYNC'),
        ('command', TOY_CREATE),
        ('command', 'TRUNCATE TABLE `toy_view_log`'),
    ]
    message = str(raised.value)
    assert 'loading toy_view into toy_view_log failed on ' in message
    assert 'release_data/view/toy_view/part-1.parquet' in message
    assert message.endswith(f'; its table: {TOY_CREATE}')
    assert isinstance(raised.value.__cause__, DatabaseError)


# on a live ClickHouse

ROWS = {'id': ['a', 'b', 'c'], 'name': ['na', 'nb', 'nc'], 'target': ['t1', None, 't2']}


def _live_task(tmp_path, client, server: tuple[str, int], rows: dict, dataset: str = 'declared_view'):
    data = tmp_path / 'release_data' / 'view' / dataset
    data.mkdir(parents=True, exist_ok=True)
    schema = {'id': pl.String, 'name': pl.String, 'target': pl.String}
    pl.DataFrame(rows, schema=schema).write_parquet(data / 'part-0.parquet')
    host, port = server
    spec = _spec(dataset, clickhouse_host=host, clickhouse_port=port, clickhouse_database=client.database)
    return _task(tmp_path, spec)


def _tables(client) -> set[str]:
    return set(client.query('SELECT name FROM system.tables WHERE database = currentDatabase()').result_columns[0])


def test_the_view_s_table_is_built_and_the_landing_dropped(tmp_path, clickhouse_database, clickhouse_server):
    live = clickhouse_database
    _run(_live_task(tmp_path, live, clickhouse_server, ROWS))
    assert live.query('SELECT id, name, target FROM declared_view ORDER BY id').result_rows == [
        ('a', 'na', 't1'), ('b', 'nb', None), ('c', 'nc', 't2'),
    ]  # fmt: skip
    _run(_live_task(tmp_path, live, clickhouse_server, ROWS, 'by_target_view'))
    assert live.query('SELECT id, target FROM by_target_view ORDER BY id').result_rows == [
        ('a', 't1'), ('b', None), ('c', 't2'),
    ]  # fmt: skip
    assert _tables(live) == {'declared_view', 'declared_names', 'by_target_view'}


def test_a_load_reruns_over_its_dictionary(tmp_path, clickhouse_database, clickhouse_server):
    live = clickhouse_database
    _run(_live_task(tmp_path, live, clickhouse_server, ROWS))
    assert live.command("SELECT dictGet('declared_names', 'name', tuple('b'))") == 'nb'
    _run(_live_task(tmp_path, live, clickhouse_server, ROWS | {'name': ['na', 'nb2', 'nc']}))
    assert live.command('SELECT count() FROM declared_view') == 3
    assert live.command("SELECT dictGet('declared_names', 'name', tuple('b'))") == 'nb2'


def test_no_count_on_a_rocksdb_table(tmp_path, clickhouse_database, clickhouse_server, monkeypatch, info):
    sent: list[str] = []
    real_connect = clickhouse_load.connect

    def recording_connect(**kw):
        client = real_connect(**kw)
        command = client.command
        monkeypatch.setattr(client, 'command', lambda sql, *a, **k: sent.append(sql) or command(sql, *a, **k))
        return client

    monkeypatch.setattr(clickhouse_load, 'connect', recording_connect)
    _run(_live_task(tmp_path, clickhouse_database, clickhouse_server, ROWS))
    counts = [sql for sql in sent if 'count()' in sql and 'throwIf' not in sql]
    assert counts == ['SELECT count() FROM `declared_view_log`']
    assert 'declared_view_log: 3 rows' in info


def test_duplicate_rocksdb_key_fails_the_load(tmp_path, clickhouse_database, clickhouse_server):
    duplicated = {'id': ['a', 'a'], 'name': ['n1', 'n2'], 'target': ['t1', 't2']}
    with pytest.raises(ClickhouseLoadError, match='duplicate id in declared_view'):
        _run(_live_task(tmp_path, clickhouse_database, clickhouse_server, duplicated))


def test_rerun_after_a_failed_build_loads_one_copy(tmp_path, clickhouse_database, clickhouse_server, monkeypatch):
    live = clickhouse_database

    def fail(client, model):
        raise ClickhouseLoadError('build failed')

    with monkeypatch.context() as m:
        m.setattr(clickhouse_load, 'build_served', fail)
        with pytest.raises(ClickhouseLoadError, match='build failed'):
            _run(_live_task(tmp_path, live, clickhouse_server, ROWS))
    assert live.command('SELECT count() FROM declared_view_log') == 3
    _run(_live_task(tmp_path, live, clickhouse_server, ROWS))
    assert live.command('SELECT count() FROM declared_view') == 3


def test_null_rocksdb_key_is_not_a_duplicate(clickhouse_database):
    clickhouse_database.command('CREATE TABLE declared_view_log (id Nullable(String)) ENGINE = Log')
    clickhouse_database.command("INSERT INTO declared_view_log VALUES (NULL), ('a')")
    clickhouse_database.command(uniqueness_sql(Declared))
    clickhouse_database.command('INSERT INTO declared_view_log VALUES (NULL)')
    with pytest.raises(DatabaseError, match='duplicate id in declared_view'):
        clickhouse_database.command(uniqueness_sql(Declared))
