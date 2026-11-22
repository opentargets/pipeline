"""clickhouse_load connects to the running server and runs each script statement by statement."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml
from clickhouse_connect.driver.exceptions import DatabaseError, OperationalError
from loguru import logger
from ot_datamodels.base import DatasetModel

from pos import clickhouse_ddl
from pos.clickhouse_ddl import ddl_for
from pos.tasks import clickhouse_load
from pos.tasks.clickhouse_load import ClickhouseLoad, ClickhouseLoadError, ClickhouseLoadSpec

ROOT = Path(__file__).parents[1]


def test_clickhouse_load_fails_on_no_parquet(tmp_path, monkeypatch):
    datasets = tmp_path / 'datasets.yaml'
    datasets.write_text('clickhouse:\n  t:\n    table: t_log\n    input_dir: output/t\n    preload_script: pre.sql\n')
    (tmp_path / 'release_data' / 'output' / 't').mkdir(parents=True)
    client = MagicMock()
    monkeypatch.setattr(clickhouse_load, 'connect', lambda **kw: client)
    spec = ClickhouseLoadSpec(
        name='clickhouse_load t', dataset='t', data_dir_parent='release_data', dataset_config_path=str(datasets)
    )
    task = ClickhouseLoad(spec, MagicMock(config=MagicMock(work_path=tmp_path)))
    with pytest.raises(ClickhouseLoadError, match=r'dataset t in .*release_data/output/t'):
        ClickhouseLoad.run.__wrapped__(task)  # ty: ignore[unresolved-attribute]
    client.query.assert_not_called()


def test_clickhouse_load_optional_runs_scripts_on_no_parquet(tmp_path, monkeypatch):
    pre = tmp_path / 'pre.sql'
    pre.write_text('CREATE TABLE t_log (x String) ENGINE = Log;\n')
    datasets = tmp_path / 'datasets.yaml'
    datasets.write_text(
        f'clickhouse:\n  t:\n    table: t_log\n    input_dir: output/t\n    optional: true\n    preload_script: {pre}\n'
    )
    client = MagicMock()
    monkeypatch.setattr(clickhouse_load, 'connect', lambda **kw: client)
    spec = ClickhouseLoadSpec(
        name='clickhouse_load t', dataset='t', data_dir_parent='release_data', dataset_config_path=str(datasets)
    )
    task = ClickhouseLoad(spec, MagicMock(config=MagicMock(work_path=tmp_path)))
    ClickhouseLoad.run.__wrapped__(task)  # ty: ignore[unresolved-attribute]
    assert [c.args[0] for c in client.query.call_args_list] == ['CREATE TABLE t_log (x String) ENGINE = Log']


def test_clickhouse_load_runs_split_statements(tmp_path, monkeypatch):
    pre = tmp_path / 'pre.sql'
    pre.write_text("-- a; comment\nCREATE TABLE t_log (x String DEFAULT ';') engine = Log;\n")
    post = tmp_path / 'post.sql'
    post.write_text('CREATE TABLE t AS SELECT * FROM t_log;\nDROP TABLE t_log SYNC;\n')
    datasets = tmp_path / 'datasets.yaml'
    datasets.write_text(
        f'clickhouse:\n  t:\n    table: t_log\n    input_dir: output/t\n'
        f'    preload_script: {pre}\n    postload_script: {post}\n'
    )
    data = tmp_path / 'release_data' / 'output' / 't'
    data.mkdir(parents=True)
    (data / 'part-0.parquet').write_bytes(b'')

    client = MagicMock()
    connected = {}
    monkeypatch.setattr(clickhouse_load, 'connect', lambda **kw: connected.update(kw) or client)
    inserted = []
    monkeypatch.setattr(
        clickhouse_load,
        'insert_file',
        lambda c, table, path, fmt, settings: inserted.append((table, path, settings)),
    )

    spec = ClickhouseLoadSpec(
        name='clickhouse_load t', dataset='t', data_dir_parent='release_data', dataset_config_path=str(datasets)
    )
    task = ClickhouseLoad(spec, MagicMock(config=MagicMock(work_path=tmp_path)))
    ClickhouseLoad.run.__wrapped__(task)  # ty: ignore[unresolved-attribute]

    assert connected['host'] == 'localhost'
    assert connected['database'] == 'ot'
    assert [c.args[0] for c in client.query.call_args_list] == [
        "-- a; comment\nCREATE TABLE t_log (x String DEFAULT ';') engine = Log",
        'CREATE TABLE t AS SELECT * FROM t_log',
        'DROP TABLE t_log SYNC',
    ]
    assert inserted == [('t_log', str(data / 'part-0.parquet'), {'async_insert': 0})]


def test_clickhouse_load_derived_runs_scripts_without_input(tmp_path, monkeypatch):
    pre = tmp_path / 'pre.sql'
    pre.write_text('CREATE TABLE d ENGINE = Log AS SELECT 1;\n')
    post = tmp_path / 'post.sql'
    post.write_text('OPTIMIZE TABLE d;\n')
    datasets = tmp_path / 'datasets.yaml'
    datasets.write_text(
        f'clickhouse:\n  d:\n    table: d\n    derived: true\n    preload_script: {pre}\n    postload_script: {post}\n'
    )
    client = MagicMock()
    monkeypatch.setattr(clickhouse_load, 'connect', lambda **kw: client)
    inserted = []
    monkeypatch.setattr(clickhouse_load, 'insert_file', lambda *a, **kw: inserted.append(a))
    spec = ClickhouseLoadSpec(
        name='clickhouse_load d', dataset='d', data_dir_parent='release_data', dataset_config_path=str(datasets)
    )
    task = ClickhouseLoad(spec, MagicMock(config=MagicMock(work_path=tmp_path)))
    ClickhouseLoad.run.__wrapped__(task)  # ty: ignore[unresolved-attribute]

    assert [c.args[0] for c in client.query.call_args_list] == [
        'CREATE TABLE d ENGINE = Log AS SELECT 1',
        'OPTIMIZE TABLE d',
    ]
    assert inserted == []


def _two_file_task(tmp_path, monkeypatch, failures):
    """A load of two files whose inserts raise the given errors in turn, then succeed."""
    datasets = tmp_path / 'datasets.yaml'
    datasets.write_text('clickhouse:\n  t:\n    table: t_log\n    input_dir: output/t\n    preload_script: pre.sql\n')
    (tmp_path / 'pre.sql').write_text('CREATE TABLE IF NOT EXISTS t_log (x String) ENGINE = Log;\n')
    monkeypatch.chdir(tmp_path)
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
    spec = ClickhouseLoadSpec(
        name='clickhouse_load t', dataset='t', data_dir_parent='release_data', dataset_config_path=str(datasets)
    )
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
    with pytest.raises(DatabaseError):
        ClickhouseLoad.run.__wrapped__(task)  # ty: ignore[unresolved-attribute]
    assert [e for e in events if e[0] == 'command'] == [('command', 'TRUNCATE TABLE `t_log`')]


# generated DDL


class Toy(DatasetModel):
    dataset_name = 'toy'
    dataset_title = 'Toy'
    dataset_description = 'Toy.'
    example = {}

    id: str


class EvidenceA(DatasetModel):
    dataset_name = 'evidence_a'
    dataset_title = 'A'
    dataset_description = 'A.'
    example = {}

    id: str
    score: float | None


class EvidenceB(DatasetModel):
    dataset_name = 'evidence_b'
    dataset_title = 'B'
    dataset_description = 'B.'
    example = {}

    id: str
    extra: str


class EvidencePpp(DatasetModel):
    dataset_name = 'evidence_ppp'
    dataset_title = 'P'
    dataset_description = 'P.'
    products = ('ppp',)
    example = {}

    id: str
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
    monkeypatch.setattr(clickhouse_ddl, 'discover_datasets', lambda: MODELS)
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


def test_credible_set_variant_uses_its_engine():
    entry = yaml.safe_load((ROOT / 'config' / 'datasets.yaml').read_text())['clickhouse']['credible_set_variant']
    ((table, sql),) = ddl_for(entry, 'platform')
    assert table == 'credible_set_variant_log'
    assert sql.endswith('ENGINE = MergeTree ORDER BY (credibleSetId, variantId)')


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


def test_evidence_retry_truncates_every_table(tmp_path, monkeypatch):
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


def test_models_matching_nothing_fail_before_connecting(tmp_path, monkeypatch):
    datasets = EVIDENCE.replace("'evidence_*'", "'evidence_none_*'")
    task, events = _generated_task(tmp_path, monkeypatch, datasets, {})
    with pytest.raises(ValueError, match=r"evidence_log: models 'evidence_none_\*' matches no model for platform"):
        _run(task)
    assert events == []


def test_scripts_log_every_statement(tmp_path, monkeypatch, info):
    pre = tmp_path / 'pre.sql'
    pre.write_text('CREATE TABLE d ENGINE = Log AS SELECT 1;\n')
    post = tmp_path / 'post.sql'
    post.write_text('CREATE TABLE e ENGINE = Log AS SELECT * FROM d;\nDROP TABLE d;\n')
    datasets = (
        f'clickhouse:\n  d:\n    table: d\n    derived: true\n    preload_script: {pre}\n    postload_script: {post}\n'
    )
    task, events = _generated_task(tmp_path, monkeypatch, datasets, {})
    _run(task)
    statements = ['CREATE TABLE d ENGINE = Log AS SELECT 1', 'CREATE TABLE e ENGINE = Log AS SELECT * FROM d',
                  'DROP TABLE d']  # fmt: skip
    assert [e[1] for e in events] == statements
    assert [line for line in info if line in statements] == statements
