"""clickhouse_load creates each landing table from its model, loads it and builds its declared tables."""

from pathlib import Path
from typing import Annotated
from unittest.mock import MagicMock

import polars as pl
import pytest
import yaml
from clickhouse_connect.driver.exceptions import DatabaseError, OperationalError
from loguru import logger
from ot_datamodels.base import DatasetModel, PrimaryKey

from pos import clickhouse_ddl
from pos.clickhouse_ddl import uniqueness_sql
from pos.clickhouse_tables import ClickHouseTable
from pos.tasks import clickhouse_load
from pos.tasks.clickhouse_load import ClickhouseLoad, ClickhouseLoadError, ClickhouseLoadSpec

ROOT = Path(__file__).parents[1]


TOY = 'clickhouse:\n  t:\n    table: t_log\n    input_dir: output/t\n    model: toy\n'
TOY_CREATE = 'CREATE TABLE IF NOT EXISTS `t_log` (`id` String) ENGINE = Log'


def _toy_spec(tmp_path, monkeypatch, datasets: str) -> ClickhouseLoadSpec:
    monkeypatch.setattr(clickhouse_ddl, 'discover_models', lambda: {'toy': Toy})
    config = tmp_path / 'datasets.yaml'
    config.write_text(datasets)
    return ClickhouseLoadSpec(
        name='clickhouse_load t', dataset='t', data_dir_parent='release_data', dataset_config_path=str(config),
        product='platform',
    )


def test_clickhouse_load_fails_on_no_parquet(tmp_path, monkeypatch):
    (tmp_path / 'release_data' / 'output' / 't').mkdir(parents=True)
    client = MagicMock()
    monkeypatch.setattr(clickhouse_load, 'connect', lambda **kw: client)
    task = ClickhouseLoad(_toy_spec(tmp_path, monkeypatch, TOY), MagicMock(config=MagicMock(work_path=tmp_path)))
    with pytest.raises(ClickhouseLoadError, match=r'dataset t in .*release_data/output/t'):
        ClickhouseLoad.run.__wrapped__(task)  # ty: ignore[unresolved-attribute]
    client.command.assert_not_called()


def test_clickhouse_load_fails_on_no_parquet_even_if_optional(tmp_path, monkeypatch):
    client = MagicMock()
    monkeypatch.setattr(clickhouse_load, 'connect', lambda **kw: client)
    spec = _toy_spec(tmp_path, monkeypatch, TOY + '    optional: true\n')
    task = ClickhouseLoad(spec, MagicMock(config=MagicMock(work_path=tmp_path)))
    with pytest.raises(ClickhouseLoadError, match=r'dataset t in .*release_data/output/t'):
        ClickhouseLoad.run.__wrapped__(task)  # ty: ignore[unresolved-attribute]
    client.command.assert_not_called()


def _two_file_task(tmp_path, monkeypatch, failures):
    """A load of two files whose inserts raise the given errors in turn, then succeed."""
    spec = _toy_spec(tmp_path, monkeypatch, TOY)
    data = tmp_path / 'release_data' / 'output' / 't'
    data.mkdir(parents=True)
    for name in ('part-0.parquet', 'part-1.parquet'):
        (data / name).write_bytes(b'')
    client = MagicMock()
    monkeypatch.setattr(clickhouse_load, 'connect', lambda **kw: client)
    events = []
    errors = list(failures)

    def insert(c, table, path, fmt, settings):
        name = path.rsplit('/', 1)[1]
        if errors and name == 'part-1.parquet':
            events.append(('failed', name))
            raise errors.pop(0)
        events.append(('insert', name))

    monkeypatch.setattr(clickhouse_load, 'insert_file', insert)
    client.command.side_effect = lambda sql: events.append(('command', sql))
    return ClickhouseLoad(spec, MagicMock(config=MagicMock(work_path=tmp_path))), events


def test_clickhouse_load_truncates_and_reloads_every_file_after_a_failed_insert(tmp_path, monkeypatch):
    task, events = _two_file_task(tmp_path, monkeypatch, [OperationalError('timed out')])
    ClickhouseLoad.run.__wrapped__(task)  # ty: ignore[unresolved-attribute]
    truncate = events.index(('command', 'TRUNCATE TABLE `t_log`'))
    assert ('failed', 'part-1.parquet') in events[:truncate]
    assert sorted(events[truncate + 1 : -1]) == [('insert', 'part-0.parquet'), ('insert', 'part-1.parquet')]
    assert events[-1] == ('command', 'SELECT count() FROM `t_log`')


def test_clickhouse_load_fails_when_the_reload_fails_too(tmp_path, monkeypatch):
    task, events = _two_file_task(tmp_path, monkeypatch, [DatabaseError('code: 159'), DatabaseError('code: 159')])
    with pytest.raises(ClickhouseLoadError) as raised:
        ClickhouseLoad.run.__wrapped__(task)  # ty: ignore[unresolved-attribute]
    assert [e for e in events if e[0] == 'command'] == [('command', TOY_CREATE), ('command', 'TRUNCATE TABLE `t_log`')]
    message = str(raised.value)
    assert 'loading t into t_log failed on ' in message
    assert 'release_data/output/t/part-1.parquet' in message
    assert message.endswith(f'; its table: {TOY_CREATE}')
    assert isinstance(raised.value.__cause__, DatabaseError)


# generated DDL


class Toy(DatasetModel):
    dataset_name = 'toy'
    dataset_title = 'Toy'
    dataset_description = 'Toy.'
    example = {}

    id: Annotated[str, PrimaryKey()]


class EvidenceA(DatasetModel):
    dataset_name = 'evidence_a'
    dataset_title = 'A'
    dataset_description = 'A.'
    example = {}

    id: Annotated[str, PrimaryKey()]
    score: float | None


class EvidenceB(DatasetModel):
    dataset_name = 'evidence_b'
    dataset_title = 'B'
    dataset_description = 'B.'
    example = {}

    id: Annotated[str, PrimaryKey()]
    extra: str


class EvidencePpp(DatasetModel):
    dataset_name = 'evidence_ppp'
    dataset_title = 'P'
    dataset_description = 'P.'
    products = ('ppp',)
    example = {}

    id: Annotated[str, PrimaryKey()]
    assays: str


MODELS = {m.dataset_name: m for m in (Toy, EvidenceA, EvidenceB, EvidencePpp)}
EVIDENCE = "clickhouse:\n  evidence:\n    table: evidence_log\n    input_dir: output\n    models: 'evidence_*'\n"
UNION_CREATE = (
    'CREATE TABLE IF NOT EXISTS `evidence_log` (`id` String, `score` Nullable(Float64), `extra` Nullable(String)) '
    'ENGINE = Log'
)
INSERT_A = 'INSERT INTO `evidence_log` (`id`, `score`) SELECT `id`, `score` FROM `evidence_a_log`'


class FakeClient:
    """Records every statement and insert in `events`; `count()` returns 7; `fail` maps a statement prefix to the
    errors it raises in turn."""

    def __init__(self, events: list, fail: dict | None = None) -> None:
        self.events = events
        self.fail = fail or {}

    def _run(self, sql: str):
        for prefix, errors in self.fail.items():
            if sql.startswith(prefix) and errors:
                self.events.append(('failed', sql))
                raise errors.pop(0)
        self.events.append(('command', sql))
        return 7 if sql.startswith('SELECT count()') else None

    command = query = _run


def _generated_task(tmp_path, monkeypatch, datasets: str, parts: dict[str, int], *, product: str = 'platform',
                    fail: dict | None = None, insert_fail: dict | None = None):  # fmt: skip
    """A load of `datasets` with `parts[dir]` parquet files under `release_data/output/<dir>`."""
    monkeypatch.setattr(clickhouse_ddl, 'discover_models', lambda: MODELS)
    tmp_path.mkdir(exist_ok=True)
    config = tmp_path / 'datasets.yaml'
    config.write_text(datasets)
    for directory, n in parts.items():
        data = tmp_path / 'release_data' / 'output' / directory
        data.mkdir(parents=True)
        for i in range(n):
            (data / f'part-{i}.parquet').write_bytes(b'')
    events: list = []
    monkeypatch.setattr(clickhouse_load, 'connect', lambda **kw: FakeClient(events, fail))
    errors = insert_fail or {}

    def insert(c, table, path, fmt, settings):
        name = path.rsplit('/', 1)[1]
        if errors.get(table):
            events.append(('failed', table, name))
            raise errors[table].pop(0)
        events.append(('insert', table, name))

    monkeypatch.setattr(clickhouse_load, 'insert_file', insert)
    dataset = yaml.safe_load(datasets)['clickhouse'].popitem()[0]
    spec = ClickhouseLoadSpec(
        name=f'clickhouse_load {dataset}', dataset=dataset, data_dir_parent='release_data',
        dataset_config_path=str(config), product=product,
    )  # fmt: skip
    return ClickhouseLoad(spec, MagicMock(config=MagicMock(work_path=tmp_path))), events


def _run(task):
    ClickhouseLoad.run.__wrapped__(task)  # ty: ignore[unresolved-attribute]


@pytest.fixture
def info(monkeypatch):
    lines: list[str] = []
    sink = logger.add(lambda m: lines.append(m.record['message']), level='INFO')
    yield lines
    logger.remove(sink)


def test_model_entry_creates_the_table_then_inserts(tmp_path, monkeypatch, info):
    datasets = 'clickhouse:\n  toy:\n    table: toy_log\n    input_dir: output/toy\n    model: toy\n'
    task, events = _generated_task(tmp_path, monkeypatch, datasets, {'toy': 2})
    _run(task)
    create = 'CREATE TABLE IF NOT EXISTS `toy_log` (`id` String) ENGINE = Log'
    assert events == [
        ('command', create),
        ('insert', 'toy_log', 'part-0.parquet'),
        ('insert', 'toy_log', 'part-1.parquet'),
        ('command', 'SELECT count() FROM `toy_log`'),
    ]
    assert create in info
    assert 'toy_log: 7 rows' in info


def test_model_entry_uses_its_engine(tmp_path, monkeypatch):
    datasets = (
        'clickhouse:\n  toy:\n    table: toy_log\n    input_dir: output/toy\n    model: toy\n'
        '    engine: MergeTree ORDER BY (id)\n'
    )
    task, events = _generated_task(tmp_path, monkeypatch, datasets, {'toy': 1})
    _run(task)
    create = 'CREATE TABLE IF NOT EXISTS `toy_log` (`id` String) ENGINE = MergeTree ORDER BY (id)'
    assert events[0] == ('command', create)


def test_evidence_loads_each_member_into_its_own_table(tmp_path, monkeypatch, info):
    task, events = _generated_task(tmp_path, monkeypatch, EVIDENCE, {'evidence_a': 2, 'evidence_b': 1})
    _run(task)
    creates = [e[1] for e in events if e[0] == 'command' and e[1].startswith('CREATE')]
    assert creates == [
        'CREATE TABLE IF NOT EXISTS `evidence_a_log` (`id` String, `score` Nullable(Float64)) ENGINE = Log',
        'CREATE TABLE IF NOT EXISTS `evidence_b_log` (`id` String, `extra` String) ENGINE = Log',
        UNION_CREATE,
    ]
    assert [e[1:] for e in events if e[0] == 'insert'] == [
        ('evidence_a_log', 'part-0.parquet'),
        ('evidence_a_log', 'part-1.parquet'),
        ('evidence_b_log', 'part-0.parquet'),
    ]
    assert set(creates) <= set(info)
    for table in ('evidence_a_log', 'evidence_b_log', 'evidence_log'):
        assert f'{table}: 7 rows' in info


EVIDENCE_DECLARED = EVIDENCE + (
    '    tables:\n'
    '      - {name: evidence, engine: rocksdb}\n'
    '      - {name: evidence_by_score, engine: mergetree, order_by: [score], columns: [score, id], '
    'where: score IS NOT NULL}\n'
)


def test_declared_union_is_built_from_the_union_after_a_key_check(tmp_path, monkeypatch):
    task, events = _generated_task(tmp_path, monkeypatch, EVIDENCE_DECLARED, {'evidence_a': 1, 'evidence_b': 1})
    _run(task)
    commands = [e[1] for e in events if e[0] == 'command' and not e[1].startswith('SELECT count()')]
    assert commands[0] == 'DROP TABLE IF EXISTS `evidence_log` SYNC'
    union = commands.index(UNION_CREATE)
    check = commands.index(
        "SELECT throwIf(count() != uniqExact(tuple(`id`)), 'duplicate id in evidence') FROM `evidence_log`"
    )
    assert union < check
    assert commands[check + 1 :] == [
        'DROP TABLE IF EXISTS `evidence` SYNC',
        (
            'CREATE TABLE `evidence` (`id` String, `score` Nullable(Float64), `extra` Nullable(String)) '
            'ENGINE = EmbeddedRocksDB() PRIMARY KEY `id`'
        ),
        'INSERT INTO `evidence` SELECT `id`, `score`, `extra` FROM `evidence_log`',
        'DROP TABLE IF EXISTS `evidence_by_score` SYNC',
        (
            'CREATE TABLE `evidence_by_score` (`score` Nullable(Float64), `id` String) '
            'ENGINE = MergeTree() ORDER BY (`score`) SETTINGS allow_nullable_key = 1'
        ),
        'INSERT INTO `evidence_by_score` SELECT `score`, `id` FROM `evidence_log` WHERE score IS NOT NULL',
        'OPTIMIZE TABLE `evidence_by_score` FINAL',
        'DROP TABLE `evidence_log` SYNC',
    ]


def test_unknown_union_declaration_fields_fail(tmp_path, monkeypatch):
    datasets = EVIDENCE + '    tables:\n      - {name: e, engine: rocksdb, key: id, optimize: true}\n'
    with pytest.raises(ValueError, match=r"evidence_log: unknown ClickHouseTable fields \['key', 'optimize'\]"):
        _generated_task(tmp_path, monkeypatch, datasets, {'evidence_a': 1, 'evidence_b': 1})


def test_evidence_insert_lists_only_member_columns(tmp_path, monkeypatch):
    task, events = _generated_task(tmp_path, monkeypatch, EVIDENCE, {'evidence_a': 1, 'evidence_b': 1})
    _run(task)
    start = events.index(('command', UNION_CREATE))
    assert events[start + 1 :] == [
        ('command', INSERT_A),
        ('command', 'INSERT INTO `evidence_log` (`id`, `extra`) SELECT `id`, `extra` FROM `evidence_b_log`'),
        ('command', 'SELECT count() FROM `evidence_log`'),
        ('command', 'DROP TABLE `evidence_a_log`'),
        ('command', 'DROP TABLE `evidence_b_log`'),
    ]


def test_missing_member_fails_unless_optional(tmp_path, monkeypatch, info):
    task, events = _generated_task(tmp_path, monkeypatch, EVIDENCE, {'evidence_a': 1})
    with pytest.raises(ClickhouseLoadError, match=r'evidence_b in .*release_data/output/evidence_b'):
        _run(task)
    assert events == []

    optional = EVIDENCE + '    optional_models: [evidence_b]\n'
    task, events = _generated_task(tmp_path / 'optional', monkeypatch, optional, {'evidence_a': 1})
    _run(task)
    tables = {e[1].split('`')[1] for e in events if e[0] == 'command' and e[1].startswith('CREATE')}
    assert tables == {'evidence_a_log', 'evidence_log'}
    assert any('evidence_b' in line and 'optional' in line for line in info)
    assert ('command', INSERT_A) in events


def test_platform_skips_ppp_models(tmp_path, monkeypatch):
    parts = {'evidence_a': 1, 'evidence_b': 1}
    task, events = _generated_task(tmp_path, monkeypatch, EVIDENCE, parts)
    _run(task)
    assert not any('evidence_ppp' in e[1] or 'assays' in e[1] for e in events)

    parts['evidence_ppp'] = 1
    task, events = _generated_task(tmp_path / 'ppp', monkeypatch, EVIDENCE, parts, product='ppp')
    _run(task)
    assert ('insert', 'evidence_ppp_log', 'part-0.parquet') in events
    assert any(e[1].startswith('CREATE TABLE IF NOT EXISTS `evidence_log`') and '`assays`' in e[1] for e in events)


def test_evidence_retry_truncates_and_reloads_only_the_failing_table(tmp_path, monkeypatch):
    parts = {'evidence_a': 2, 'evidence_b': 1}
    task, events = _generated_task(tmp_path, monkeypatch, EVIDENCE, parts,
                                   insert_fail={'evidence_b_log': [OperationalError('timed out')]})  # fmt: skip
    _run(task)
    truncates = [e[1] for e in events if e[0] == 'command' and e[1].startswith('TRUNCATE')]
    assert truncates == ['TRUNCATE TABLE `evidence_b_log`']
    at = events.index(('command', 'TRUNCATE TABLE `evidence_b_log`'))
    assert events[at - 1] == ('failed', 'evidence_b_log', 'part-0.parquet')
    assert events[at + 1] == ('insert', 'evidence_b_log', 'part-0.parquet')

    union = 'INSERT INTO `evidence_log` (`id`, `extra`)'
    task, events = _generated_task(tmp_path / 'union', monkeypatch, EVIDENCE, parts,
                                   fail={union: [OperationalError('timed out')]})  # fmt: skip
    _run(task)
    at = events.index(('command', 'TRUNCATE TABLE `evidence_log`'))
    assert events[at - 1][0] == 'failed'
    assert [e[1].split(' FROM ')[-1] for e in events[at + 1 : at + 3]] == ['`evidence_a_log`', '`evidence_b_log`']
    assert events[at + 3] == ('command', 'SELECT count() FROM `evidence_log`')

    task, events = _generated_task(tmp_path / 'twice', monkeypatch, EVIDENCE, parts,
                                   fail={union: [OperationalError('a'), OperationalError('b')]})  # fmt: skip
    with pytest.raises(OperationalError):
        _run(task)
    assert not any(e[1].startswith('DROP') for e in events)


def test_failing_member_file_names_its_source_and_types(tmp_path, monkeypatch):
    parts = {'evidence_a': 1, 'evidence_b': 1}
    errors = [DatabaseError('code: 70'), DatabaseError('code: 70')]
    task, _ = _generated_task(tmp_path, monkeypatch, EVIDENCE, parts, insert_fail={'evidence_b_log': errors})
    create = 'CREATE TABLE IF NOT EXISTS `evidence_b_log` (`id` String, `extra` String) ENGINE = Log'
    with pytest.raises(ClickhouseLoadError) as raised:
        _run(task)
    message = str(raised.value)
    assert 'loading evidence_b into evidence_b_log failed on ' in message
    assert 'release_data/output/evidence_b/part-0.parquet' in message
    assert create in message


def test_failing_model_file_names_its_source_and_types(tmp_path, monkeypatch):
    datasets = 'clickhouse:\n  toy:\n    table: toy_log\n    input_dir: output/toy\n    model: toy\n'
    errors = [DatabaseError('code: 70'), DatabaseError('code: 70')]
    task, _ = _generated_task(tmp_path, monkeypatch, datasets, {'toy': 1}, insert_fail={'toy_log': errors})
    with pytest.raises(ClickhouseLoadError) as raised:
        _run(task)
    message = str(raised.value)
    assert 'loading toy into toy_log failed on ' in message
    assert 'release_data/output/toy/part-0.parquet' in message
    assert 'CREATE TABLE IF NOT EXISTS `toy_log` (`id` String) ENGINE = Log' in message


def test_models_matching_nothing_fail_before_connecting(tmp_path, monkeypatch):
    datasets = EVIDENCE.replace("'evidence_*'", "'evidence_none_*'")
    task, events = _generated_task(tmp_path, monkeypatch, datasets, {})
    with pytest.raises(ValueError, match=r"evidence_log: models 'evidence_none_\*' matches no model for platform"):
        _run(task)
    assert events == []


# declared tables, on a live ClickHouse

class Declared(DatasetModel):
    dataset_name = 'declared'
    dataset_title = 'Declared'
    dataset_description = 'Declared.'
    example = {}

    id: Annotated[str, PrimaryKey()]
    name: str
    target: str | None


DECLARED = """\
clickhouse:
  declared:
    table: declared_log
    input_dir: output/declared
    model: declared
    tables:
      - {name: declared_by_id, engine: rocksdb}
      - name: declared_by_target
        engine: mergetree
        order_by: [target, id]
        columns: [id, target]
        where: target IS NOT NULL
"""
# a dictionary reads its entry's one table
NAMED = """\
clickhouse:
  declared:
    table: declared_log
    input_dir: output/declared
    model: declared
    tables:
      - {name: declared_by_id, engine: rocksdb}
    dictionaries:
      - {name: declared_names, key: id, attributes: [name]}
"""
ROWS = {'id': ['a', 'b', 'c'], 'name': ['na', 'nb', 'nc'], 'target': ['t1', None, 't2']}


@pytest.fixture
def live(monkeypatch, clickhouse_database):
    """A client on a fresh database of the test server; `Declared` discoverable."""
    monkeypatch.setattr(clickhouse_ddl, 'discover_models', lambda: {'declared': Declared})
    return clickhouse_database


def _declared_task(tmp_path, client, server: tuple[str, int], rows: dict, yaml: str = DECLARED) -> ClickhouseLoad:
    config = tmp_path / 'datasets.yaml'
    config.write_text(yaml)
    data = tmp_path / 'release_data' / 'output' / 'declared'
    data.mkdir(parents=True, exist_ok=True)
    schema = {'id': pl.String, 'name': pl.String, 'target': pl.String}
    pl.DataFrame(rows, schema=schema).write_parquet(data / 'part-0.parquet')
    host, port = server
    spec = ClickhouseLoadSpec(
        name='clickhouse_load declared', dataset='declared', data_dir_parent='release_data',
        dataset_config_path=str(config), product='platform', clickhouse_host=host, clickhouse_port=port,
        clickhouse_database=client.database,
    )  # fmt: skip
    return ClickhouseLoad(spec, MagicMock(config=MagicMock(work_path=tmp_path)))


def _tables(client) -> set[str]:
    return set(client.query('SELECT name FROM system.tables WHERE database = currentDatabase()').result_columns[0])


def test_declared_tables_are_built_and_the_landing_dropped(tmp_path, live, clickhouse_server):
    _run(_declared_task(tmp_path, live, clickhouse_server, ROWS))
    assert live.query('SELECT id, name, target FROM declared_by_id ORDER BY id').result_rows == [
        ('a', 'na', 't1'), ('b', 'nb', None), ('c', 'nc', 't2'),
    ]  # fmt: skip
    assert live.query('SELECT id, target FROM declared_by_target ORDER BY id').result_rows == [
        ('a', 't1'), ('c', 't2'),
    ]  # fmt: skip
    assert 'declared_log' not in _tables(live)


def test_declared_load_reruns_over_its_dictionary(tmp_path, live, clickhouse_server):
    _run(_declared_task(tmp_path, live, clickhouse_server, ROWS, NAMED))
    assert live.command("SELECT dictGet('declared_names', 'name', tuple('b'))") == 'nb'
    _run(_declared_task(tmp_path, live, clickhouse_server, ROWS | {'name': ['na', 'nb2', 'nc']}, NAMED))
    assert live.command('SELECT count() FROM declared_by_id') == 3
    assert live.command("SELECT dictGet('declared_names', 'name', tuple('b'))") == 'nb2'


def test_no_count_on_a_rocksdb_table(tmp_path, live, clickhouse_server, monkeypatch, info):
    sent: list[str] = []
    real_connect = clickhouse_load.connect

    def recording_connect(**kw):
        client = real_connect(**kw)
        command = client.command
        monkeypatch.setattr(client, 'command', lambda sql, *a, **k: sent.append(sql) or command(sql, *a, **k))
        return client

    monkeypatch.setattr(clickhouse_load, 'connect', recording_connect)
    _run(_declared_task(tmp_path, live, clickhouse_server, ROWS))
    counts = [sql for sql in sent if 'count()' in sql and 'throwIf' not in sql]
    assert counts == ['SELECT count() FROM `declared_log`', 'SELECT count() FROM `declared_by_target`']
    assert 'declared_log: 3 rows' in info
    assert 'declared_by_target: 2 rows' in info


def test_duplicate_rocksdb_key_fails_the_load(tmp_path, live, clickhouse_server):
    duplicated = {'id': ['a', 'a'], 'name': ['n1', 'n2'], 'target': ['t1', 't2']}
    with pytest.raises(ClickhouseLoadError, match='duplicate id in declared_by_id'):
        _run(_declared_task(tmp_path, live, clickhouse_server, duplicated))


@pytest.mark.parametrize(
    ('source', 'problem'),
    [
        ('', 'has neither model nor models'),
        ("    model: declared\n    models: 'declared*'\n", 'has both model and models'),
    ],
)
def test_an_entry_needs_one_of_model_and_models_at_spec_time(tmp_path, monkeypatch, source, problem):
    monkeypatch.setattr(clickhouse_ddl, 'discover_models', lambda: {'declared': Declared})
    config = tmp_path / 'datasets.yaml'
    config.write_text(DECLARED.replace('    model: declared\n', source))
    spec = ClickhouseLoadSpec(
        name='clickhouse_load declared', dataset='declared', data_dir_parent='release_data',
        dataset_config_path=str(config), product='platform',
    )  # fmt: skip
    with pytest.raises(ClickhouseLoadError, match=f'^declared: {problem}$'):
        ClickhouseLoad(spec, MagicMock(config=MagicMock(work_path=tmp_path)))


def test_rerun_after_a_failed_build_loads_one_copy(tmp_path, live, clickhouse_server, monkeypatch):
    def fail(self, client):
        raise ClickhouseLoadError('build failed')

    with monkeypatch.context() as m:
        m.setattr(ClickhouseLoad, '_build_declared', fail)
        with pytest.raises(ClickhouseLoadError, match='build failed'):
            _run(_declared_task(tmp_path, live, clickhouse_server, ROWS))
    assert live.command('SELECT count() FROM declared_log') == 3
    _run(_declared_task(tmp_path, live, clickhouse_server, ROWS))
    assert live.command('SELECT count() FROM declared_by_id') == 3
    assert live.command('SELECT count() FROM declared_by_target') == 2


def test_null_rocksdb_key_is_not_a_duplicate(clickhouse_database):
    table = ClickHouseTable('t', 'rocksdb')
    clickhouse_database.command('CREATE TABLE t_log (k Nullable(String)) ENGINE = Log')
    clickhouse_database.command("INSERT INTO t_log VALUES (NULL), ('a')")
    clickhouse_database.command(uniqueness_sql(table, 'k', 't_log'))
    clickhouse_database.command('INSERT INTO t_log VALUES (NULL)')
    with pytest.raises(DatabaseError, match='duplicate k in t'):
        clickhouse_database.command(uniqueness_sql(table, 'k', 't_log'))


class DeclaredPpp(DatasetModel):
    dataset_name = 'declared'
    dataset_title = 'Declared'
    dataset_description = 'Declared.'
    example = {}
    products = ('ppp',)

    id: Annotated[str, PrimaryKey()]


def test_declarations_building_nothing_for_the_product_fail_at_spec_time(tmp_path, monkeypatch):
    monkeypatch.setattr(clickhouse_ddl, 'discover_models', lambda: {'declared': DeclaredPpp})
    config = tmp_path / 'datasets.yaml'
    config.write_text(DECLARED)
    spec = ClickhouseLoadSpec(
        name='clickhouse_load declared', dataset='declared', data_dir_parent='release_data',
        dataset_config_path=str(config), product='platform',
    )  # fmt: skip
    with pytest.raises(ClickhouseLoadError, match='declared: declares no ClickHouse table for platform'):
        ClickhouseLoad(spec, MagicMock(config=MagicMock(work_path=tmp_path)))
    ClickhouseLoad(spec.model_copy(update={'product': 'ppp'}), MagicMock(config=MagicMock(work_path=tmp_path)))
