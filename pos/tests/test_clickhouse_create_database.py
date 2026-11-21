"""clickhouse_create_database refuses a database that already has tables, before any load."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import yaml

from pos.tasks import clickhouse_create_database as module
from pos.tasks.clickhouse_create_database import (
    ClickhouseCreateDatabase,
    ClickhouseCreateDatabaseError,
    ClickhouseCreateDatabaseSpec,
)

ROOT = Path(__file__).parents[1]
BACKEND = yaml.safe_load((ROOT / 'config' / 'config.yaml').read_text())['steps']['backend_clickhouse']


class FakeClient:
    def __init__(self, tables: list[str]) -> None:
        self.tables = tables
        self.queries: list[tuple[str, dict]] = []

    def query(self, query, parameters):
        self.queries.append((query, parameters))
        rows = [(t,) for t in self.tables] if 'system.tables' in query else []
        return SimpleNamespace(result_rows=rows)


def _run(monkeypatch, tables: list[str]) -> FakeClient:
    client = FakeClient(tables)
    monkeypatch.setattr(module, 'connect', lambda **kw: client)
    spec = ClickhouseCreateDatabaseSpec(
        name='clickhouse_create_database x', clickhouse_database='platform2612', exist_ok=True
    )
    task = ClickhouseCreateDatabase(spec, MagicMock())
    ClickhouseCreateDatabase.run.__wrapped__(task)  # ty: ignore[unresolved-attribute]
    return client


def test_empty_database_passes(monkeypatch):
    client = _run(monkeypatch, [])
    (create, listing) = client.queries
    assert create == ('CREATE DATABASE IF NOT EXISTS {database:Identifier}', {'database': 'platform2612'})
    assert listing[1] == {'database': 'platform2612'}


def test_database_with_tables_fails(monkeypatch):
    tables = [f't{i:02}' for i in range(12)]
    with pytest.raises(ClickhouseCreateDatabaseError, match=r'12 tables \(t00, .*t09, \.\.\.\)') as e:
        _run(monkeypatch, tables)
    assert "wipe that server's data disk" in str(e.value)


def test_guards_run_before_any_load():
    tasks = {t['name']: t for t in BACKEND}
    guards = [n for n in tasks if n.split(' ')[0] == 'clickhouse_create_database']
    assert len(guards) == 1
    (create_db,) = guards
    assert not tasks[create_db].get('requires')
    for name, task in tasks.items():
        if name.startswith('clickhouse_load') or name == 'explode load all datasets':
            # top-level loads require the database task directly or through another load
            assert create_db in task.get('requires', []) or any(
                r.startswith('clickhouse_load') for r in task.get('requires', [])
            ), name
