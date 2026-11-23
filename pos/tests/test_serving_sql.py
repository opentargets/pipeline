"""The serving SQL runs against the generated landing tables, empty, on a live ClickHouse.

`POS_TEST_CLICKHOUSE=<host>:<port>` (default `localhost:8123`) names the server; with no server the test skips,
unless `POS_REQUIRE_CLICKHOUSE=1`.
"""

import csv
import datetime
import json
import os
import urllib.request
import uuid
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from graphlib import CycleError, TopologicalSorter
from pathlib import Path

import polars as pl
import pytest
import yaml
from clickhouse_connect.driver.exceptions import ClickHouseError
from clickhouse_connect.driver.tools import insert_file
from ot_datamodels.discovery import discover_models
from ot_datamodels.polars import to_polars_schema

from pos.clickhouse_ddl import canonical, columns, create_table, ddl_for, member_table, members, parse_type, type_at
from pos.services.clickhouse import connect
from pos.sql import split_statements
from pos.tasks.clickhouse_load import INSERT_SETTINGS, member_drops, union_inserts

ROOT = Path(__file__).parents[1]
STEP = yaml.safe_load((ROOT / 'config' / 'config.yaml').read_text())['steps']['backend_clickhouse']
DATASETS = yaml.safe_load((ROOT / 'config' / 'datasets.yaml').read_text())['clickhouse']
with (ROOT / 'docs' / 'phase5_sql_edits.csv').open(newline='') as _fh:
    SQL_EDITS = list(csv.DictReader(_fh))
with (ROOT / 'docs' / 'phase5_cast_back.csv').open(newline='') as _fh:
    CAST_BACK = list(csv.DictReader(_fh))
with (ROOT / 'docs' / 'phase5_derived_changes.csv').open(newline='') as _fh:
    DERIVED = list(csv.DictReader(_fh))
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


@contextmanager
def _database() -> Generator:
    """A client on a fresh database of the test server, dropped afterwards."""
    host, port = SERVER.rsplit(':', 1)
    if not _reachable(host, int(port)):
        if REQUIRE:
            pytest.fail(f'no ClickHouse at {SERVER} and POS_REQUIRE_CLICKHOUSE=1')
        pytest.skip(f'no ClickHouse at {SERVER} (set POS_TEST_CLICKHOUSE, or POS_REQUIRE_CLICKHOUSE=1 to fail)')
    database = f'pos_test_{uuid.uuid4().hex}'
    admin = connect(host=host, port=int(port))
    admin.command(f'CREATE DATABASE {database}')
    try:
        yield connect(host=host, port=int(port), database=database)
    finally:
        admin.command(f'DROP DATABASE IF EXISTS {database} SYNC')


@pytest.mark.parametrize('product', ['platform', 'ppp'])
def test_serving_sql(product):
    with _database() as client:
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
        for entry in SQL_EDITS + CAST_BACK:
            column = entry['path'].split('/', 1)[0].removesuffix('[]')
            served = client.query(query, parameters={'d': client.database, 't': entry['table'], 'c': column})
            assert len(served.result_rows) == 1, f'{entry}: no column {column}'
            found = type_at([(column, parse_type(served.result_rows[0][0]))], entry['path'])
            assert found is not None, f'{entry}: no {entry["path"]}'
            assert canonical(found) == canonical(parse_type(entry['served'])), str(entry)
        for entry in DERIVED:
            served = client.query(query, parameters={'d': client.database, 't': entry['table'], 'c': entry['field']})
            assert len(served.result_rows) == 1, f'phase5_derived_changes.csv: {entry}: no column'
            found = parse_type(served.result_rows[0][0])
            assert canonical(found) == canonical(parse_type(entry['served'])), f'phase5_derived_changes.csv: {entry}'


def _as_strings(dtype):
    """`dtype` with every Enum a String: the dtype a writer has before pts conforms it."""
    if isinstance(dtype, pl.Enum):
        return pl.String()
    if isinstance(dtype, pl.List):
        return pl.List(_as_strings(dtype.inner))
    if isinstance(dtype, pl.Struct):
        return pl.Struct([pl.Field(f.name, _as_strings(f.dtype)) for f in dtype.fields])
    return dtype


SCALARS = {
    pl.String: 'x', pl.Boolean: True, pl.Int32: 1, pl.Int64: 1, pl.Float32: 0.5, pl.Float64: 0.5,
    pl.Date: datetime.date(2026, 1, 1),
}  # fmt: skip


def _value(dtype, pick: int):
    """A value of `dtype`: an Enum gets its `pick`-th label, a Boolean `pick == 0`, a list two elements (first and
    last label)."""
    if isinstance(dtype, pl.Enum):
        return dtype.categories[pick]
    if isinstance(dtype, pl.Boolean):
        return pick == 0
    if isinstance(dtype, pl.List):
        return [_value(dtype.inner, 0), _value(dtype.inner, -1)]
    if isinstance(dtype, pl.Struct):
        return {f.name: _value(f.dtype, pick) for f in dtype.fields}
    return SCALARS[type(dtype)]


def _labels(value, dtype, path: str) -> Iterator[tuple[str, object]]:
    """(path, value) of every Enum and Boolean value in `value`."""
    if isinstance(dtype, (pl.Enum, pl.Boolean)):
        yield path, value
    elif isinstance(dtype, pl.List):
        for i, v in enumerate(value):
            yield from _labels(v, dtype.inner, f'{path}[{i}]')
    elif isinstance(dtype, pl.Struct):
        for f in dtype.fields:
            yield from _labels(value[f.name], f.dtype, f'{path}/{f.name}')


#: Models with an Enum at the top level, in a list and in a struct (target_view: and a Bool in a list of structs),
#: and the dataset each loads into.
ENUM_MODELS = {
    'clinical_report': 'clinical_report',
    'variant': 'variants',
    'interaction': 'interaction',
    'evidence_gene_burden': 'evidence',
    'target_view': 'targets',
}


@pytest.mark.parametrize('model_name', sorted(ENUM_MODELS))
def test_enum_parquet_loads_into_enum8(model_name, tmp_path):
    """Polars-written Enum parquet (dictionary-encoded strings) loads into the generated Enum8 columns, labels kept;
    Booleans load into Bool."""
    model = discover_models()[model_name]
    schema = to_polars_schema(model)
    rows = [{n: _value(t, pick) for n, t in schema.items()} for pick in (0, -1)]
    strings = pl.DataFrame(rows, schema={n: _as_strings(t) for n, t in schema.items()})
    frame = strings.cast(dict(schema.items()), strict=True)  # what pts' conform does to an output
    file = tmp_path / 'part.parquet'
    frame.write_parquet(file)
    entry = DATASETS[ENUM_MODELS[model_name]]
    if 'models' in entry:
        table, create = member_table(model), create_table(member_table(model), columns(model))
    else:
        ((table, create),) = ddl_for(entry, 'platform')
    with _database() as client:
        client.command(create)
        insert_file(client, table, str(file), fmt='Parquet', settings=INSERT_SETTINGS)
        raw = client.raw_query('SELECT * FROM {t:Identifier} FORMAT JSONEachRow', parameters={'t': table}).decode()
    back = [json.loads(line) for line in raw.splitlines()]
    assert len(back) == len(rows)
    for want, got in zip(frame.to_dicts(), back, strict=True):
        for name, dtype in schema.items():
            assert list(_labels(got[name], dtype, name)) == list(_labels(want[name], dtype, name)), name
