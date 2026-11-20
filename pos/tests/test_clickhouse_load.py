"""clickhouse_load connects to the running server and runs each script statement by statement."""

from unittest.mock import MagicMock

import pytest

from pos.tasks import clickhouse_load
from pos.tasks.clickhouse_load import ClickhouseLoad, ClickhouseLoadError, ClickhouseLoadSpec


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
    monkeypatch.setattr(clickhouse_load, 'insert_file', lambda c, table, path, fmt: inserted.append((table, path)))

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
    assert inserted == [('t_log', str(data / 'part-0.parquet'))]


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
