"""The serving SQL runs against the generated landing tables, empty, on a live ClickHouse.

`POS_TEST_CLICKHOUSE=<host>:<port>` (default `localhost:8123`) names the server; with no server the test skips,
unless `POS_REQUIRE_CLICKHOUSE=1`.
"""

import csv
import os
import urllib.request
import uuid
from graphlib import CycleError, TopologicalSorter
from pathlib import Path

import pytest
import yaml
from clickhouse_connect.driver.exceptions import ClickHouseError

from pos.clickhouse_ddl import ddl_for, members
from pos.services.clickhouse import connect
from pos.sql import split_statements
from pos.tasks.clickhouse_load import member_drops, union_inserts

ROOT = Path(__file__).parents[1]
STEP = yaml.safe_load((ROOT / 'config' / 'config.yaml').read_text())['steps']['backend_clickhouse']
DATASETS = yaml.safe_load((ROOT / 'config' / 'datasets.yaml').read_text())['clickhouse']
with (ROOT / 'docs' / 'phase5_sql_edits.csv').open(newline='') as _fh:
    SQL_EDITS = list(csv.DictReader(_fh))
# read at import: the autouse fixture clears POS_* before each test
SERVER = os.environ.get('POS_TEST_CLICKHOUSE', 'localhost:8123')
REQUIRE = os.environ.get('POS_REQUIRE_CLICKHOUSE') == '1'


def _reachable(host: str, port: int) -> bool:
    try:
        with urllib.request.urlopen(f'http://{host}:{port}/ping', timeout=2) as r:
            return r.read().decode().strip() == 'Ok.'
    except OSError:
        return False


def _tasks(product: str) -> list[dict]:
    """The step's tasks for the product, explode children expanded, in `requires:` order."""
    by_name: dict[str, dict] = {}
    for task in STEP:
        if product not in task.get('products', [product]):
            continue
        if 'foreach' in task:
            for each in task['foreach']:
                for child in task['do']:
                    expanded = yaml.safe_load(yaml.safe_dump(child).replace('${each}', each))
                    by_name[expanded['name']] = expanded | {'requires': task.get('requires', [])}
        else:
            by_name[task['name']] = task
    graph = {}
    for name, task in by_name.items():
        unknown = [r for r in task.get('requires', []) if r not in by_name]
        assert not unknown, f'{name}: unknown requirement {unknown}'
        graph[name] = task.get('requires', [])
    try:
        order = list(TopologicalSorter(graph).static_order())
    except CycleError as e:
        pytest.fail(f'requires: cycle {e.args[1]}')
    return [by_name[n] for n in order]


def _statements(dataset: str, product: str) -> list[tuple[str, str]]:
    """(script, statement) a load of the dataset runs, tables empty."""
    entry = DATASETS[dataset]
    out = []
    if 'models' in entry:
        ddl = ddl_for(entry, product)
        out += [('generated DDL', sql) for _, sql in ddl]
        loaded = members(entry['models'], product)
        out += [('member inserts', sql) for sql in union_inserts(entry['table'], loaded)]
        out += [('member drops', sql) for sql in member_drops(loaded)]
    elif ddl := ddl_for(entry, product):
        out += [('generated DDL', sql) for _, sql in ddl]
    else:
        script = entry['preload_script']
        out += [(script, sql) for sql in split_statements((ROOT / script).read_text())]
    if script := entry.get('postload_script'):
        out += [(script, sql) for sql in split_statements((ROOT / script).read_text())]
    return out


@pytest.mark.parametrize('product', ['platform', 'ppp'])
def test_serving_sql(product):
    host, port = SERVER.rsplit(':', 1)
    if not _reachable(host, int(port)):
        if REQUIRE:
            pytest.fail(f'no ClickHouse at {SERVER} and POS_REQUIRE_CLICKHOUSE=1')
        pytest.skip(f'no ClickHouse at {SERVER} (set POS_TEST_CLICKHOUSE, or POS_REQUIRE_CLICKHOUSE=1 to fail)')
    database = f'pos_test_{uuid.uuid4().hex}'
    admin = connect(host=host, port=int(port))
    admin.command(f'CREATE DATABASE {database}')
    try:
        client = connect(host=host, port=int(port), database=database)
        for task in _tasks(product):
            if 'dataset' not in task:
                continue
            for script, sql in _statements(task['dataset'], product):
                try:
                    client.command(sql)
                except ClickHouseError as e:
                    pytest.fail(f'{task["name"]}: {script} failed: {e}\nstatement: {sql}')
        query = (
            'SELECT type FROM system.columns WHERE database = {d:String} AND table = {t:String} AND name = {c:String}'
        )
        for edit in SQL_EDITS:
            served = client.query(query, parameters={'d': database, 't': edit['table'], 'c': edit['field']})
            assert served.result_rows == [(edit['served'],)], f'phase5_sql_edits.csv: {edit}'
    finally:
        admin.command(f'DROP DATABASE IF EXISTS {database} SYNC')
