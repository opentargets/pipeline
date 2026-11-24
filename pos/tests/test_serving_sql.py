"""The serving SQL runs against the generated landing tables, empty, on a live ClickHouse.

`POS_TEST_CLICKHOUSE=<host>:<port>` (default `localhost:8123`) names the server; with no server the test skips,
unless `POS_REQUIRE_CLICKHOUSE=1`.
"""

import csv
import datetime
import json
import os
import re
import urllib.request
import uuid
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from graphlib import CycleError, TopologicalSorter
from io import BytesIO
from pathlib import Path

import polars as pl
import pytest
import yaml
from clickhouse_connect.driver.exceptions import ClickHouseError
from clickhouse_connect.driver.tools import insert_file
from ot_datamodels.discovery import discover_models
from ot_datamodels.outputs.disease import Synonyms
from ot_datamodels.polars import to_polars_schema

from pos.clickhouse_ddl import (
    canonical,
    columns,
    create_table,
    ddl_for,
    declarations,
    declared_sql,
    entry_key,
    is_declared,
    landing_columns,
    member_table,
    members,
    parse_type,
    render,
    type_at,
    uniqueness_sql,
)
from pos.services.clickhouse import connect
from pos.sql import split_statements
from pos.tasks.clickhouse_load import INSERT_SETTINGS, ClickhouseLoad, ClickhouseLoadError, member_drops, union_inserts

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
    """The step's tasks for the product, explode children expanded, in `requires:` order; requiring an explode
    requires its children."""
    by_name: dict[str, dict] = {}
    children: dict[str, list[str]] = {}
    for task in STEP:
        if product not in task.get('products', [product]):
            continue
        if 'foreach' in task:
            children[task['name']] = []
            for each in task['foreach']:
                for child in task['do']:
                    expanded = yaml.safe_load(yaml.safe_dump(child).replace('${each}', each))
                    by_name[expanded['name']] = expanded | {'requires': task.get('requires', [])}
                    children[task['name']].append(expanded['name'])
        else:
            by_name[task['name']] = task
    graph = {}
    for name, task in by_name.items():
        unknown = [r for r in task.get('requires', []) if r not in by_name and r not in children]
        assert not unknown, f'{name}: unknown requirement {unknown}'
        graph[name] = [c for r in task.get('requires', []) for c in children.get(r, [r])]
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
    out += [('declared tables', sql) for sql in declared_sql(entry, product)]
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


#: Models with an Enum at the top level, in a list and in a struct (target_view: and a Bool in a list of structs;
#: interaction_view: in a list of structs), and the dataset each loads into.
ENUM_MODELS = {
    'clinical_report': 'clinical_report',
    'variant': 'variants',
    'interaction_view': 'interaction',
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


def test_no_script_reads_a_declared_landing_table():
    """A declared load drops its landing table, so no remaining script may read it."""
    landings = {e['table'] for e in DATASETS.values() if is_declared(e)}
    scripts = {e[k] for e in DATASETS.values() for k in ('preload_script', 'postload_script') if k in e}
    assert landings
    reads = [
        (script, table)
        for script in sorted(scripts)
        for table in sorted(landings)
        if re.search(rf'\b{table}\b', (ROOT / script).read_text())
    ]
    assert not reads, f'scripts reading a declared landing table: {reads}'


#: The datasets whose storage-only scripts became declarations; the scripts are frozen under BEFORE.
DECLARED = (
    'association_time_series', 'baseline_expression', 'biosample', 'clinical_indication', 'clinical_report',
    'clinical_target', 'disease', 'drug', 'enhancer_to_gene', 'evidence', 'go', 'hpo', 'literature_entity_lut',
    'otar_projects', 'so', 'target_essentiality', 'w2v',
)  # fmt: skip
BEFORE = ROOT / 'tests' / 'fixtures' / 'serving_scripts_before'
#: What the scripts left that nothing reads: a view the dictionary's EXPRESSION replaces, a landing table never dropped;
#: `disease_log`, left for an aotf.sql that read it.
NOT_REBUILT = {'biosample_names_source', 'disease_log', 'otar_projects_log'}
#: Declared datasets a product no longer loads: `otar_projects` was built empty on Platform, its model is PPP-only.
NOT_LOADED = {'platform': {'association_time_series', 'otar_projects'}, 'ppp': set()}
#: Engine settings the declarations derive rather than copy. The primary key is the sort key:
PRIMARY_KEY_IS_SORT_KEY = {'clinical_indication_disease', 'clinical_indication_drug', 'clinical_target', 'ml_w2v'}
#: `allow_nullable_key` only where an `order_by` column is nullable:
NULLABLE_KEY_DROPPED = {'evidence_by_disease_and_target'}
# Every MergeTree table is optimised, `biosample_mergetree` too: the rows compared are the same.
#: Scaffolding the script left unsorted (`ORDER BY tuple()`); credible_sets_build_cleanup drops it before release.
RESORTED = {'biosample_mergetree', 'disease_mergetree'}
#: Fixture rows per landing table; generateRandom makes a few percent of a Nullable column NULL.
FIXTURE_ROWS = 300
#: The `IS NOT NULL` filters on columns the landing table has not Nullable.
NON_NULL_FILTERS = (
    ('clinical_target', 'targetId'),
    ('clinical_indication_drug', 'drugId'),
    ('clinical_indication_disease', 'diseaseId'),
    ('evidence_by_disease_and_target', 'diseaseId'),
    ('evidence_by_disease_and_target', 'targetId'),
)  # fmt: skip
#: The scripts' RocksDB keys, made unique in the fixture rows.
KEYS = {'biosampleId', 'efo_id', 'id', 'pharmacogenomicsId', 'targetId'}
#: Landing columns pts now writes that a frozen script computed: the landing it read lacked them.
SCRIPT_COMPUTED = {'drug_log': {'pharmacogenomicsIds'}, 'pharmacogenomics_log': {'pharmacogenomicsId'}}
#: Served tables a frozen script reads besides its landing table: built by their own frozen scripts, then dropped.
SCRIPT_READS = {'drug': ['pharmacogenomics']}
#: Landing tables a frozen script reads besides its own: landed, then dropped.
SCRIPT_LANDINGS = {'disease': ['studies']}
#: The model of the landing a frozen script read, where the declaration lands a view.
SCRIPT_MODEL = {'disease': 'disease'}
#: Declared tables compared by structure and row count only: pts computes their content, checked in pts
#: tests/test_drug_view.py and tests/test_disease_view.py and on real data by the release parity run.
CONTENT_FROM_PTS = {'drug', 'disease', 'disease_mergetree', 'dictionary disease_names'}


def _unique(name: str, dtype, seed: int) -> str:
    if render(dtype).startswith('UInt'):
        return f'rowNumberInAllBlocks() AS `{name}`'
    return f"concat('{seed}-', toString(rowNumberInAllBlocks()), `{name}`) AS `{name}`"


def _fill(client, table: str, cols, seed: int, overrides: dict[str, str] | None = None) -> None:
    """`FIXTURE_ROWS` rows into `table`, the same for the same `seed` on every database; `overrides` replaces a
    column with an expression over the random row."""
    overrides = overrides or {}
    structure = ', '.join(f'`{name}` {render(t)}' for name, t in cols)
    literal = structure.replace('\\', '\\\\').replace("'", "\\'")
    unique = [_unique(n, t, seed) for n, t in cols if n in KEYS and n not in overrides]
    unique += [f'{expr} AS `{name}`' for name, expr in overrides.items()]
    replace = f' REPLACE ({", ".join(unique)})' if unique else ''
    client.command(
        f"INSERT INTO `{table}` SELECT *{replace} FROM generateRandom('{literal}', {seed}, 12, 3) "  # noqa: S608 generated names
        f'LIMIT {FIXTURE_ROWS}'
    )


def _script_columns(entry, product: str) -> list:
    """The columns of the landing table the entry's frozen script read."""
    cols = landing_columns(entry, product)
    assert cols is not None, entry['table']
    return [c for c in cols if c[0] not in SCRIPT_COMPUTED.get(entry['table'], set())]


def _land(client, dataset: str, product: str, *, script: bool = False) -> None:
    """The dataset's landing table, filled with fixture rows; for `script`, the one its frozen script read."""
    entry = DATASETS[dataset]
    if script and dataset in SCRIPT_MODEL:
        entry = {'table': entry['table'], 'model': SCRIPT_MODEL[dataset]}
    if script and entry['table'] in SCRIPT_COMPUTED:
        cols = _script_columns(entry, product)
        client.command(create_table(entry['table'], cols))
        _fill(client, entry['table'], cols, 1)
        return
    for _, sql in ddl_for(entry, product):
        client.command(sql)
    if 'models' not in entry:
        _fill(client, entry['table'], landing_columns(entry, product), 1)
        return
    loaded = members(entry['models'], product)
    for seed, model in enumerate(loaded, 1):
        _fill(client, member_table(model), columns(model), seed)
    for sql in union_inserts(entry['table'], loaded) + member_drops(loaded):
        client.command(sql)


def _shape(client) -> dict:
    """Every table's engine, keys and columns, every dictionary's structure, and the rows of both."""
    d = {'d': client.database}
    tables = client.query(
        'SELECT name, engine, engine_full, sorting_key, primary_key FROM system.tables '
        "WHERE database = {d:String} AND engine != 'Dictionary'",
        parameters=d,
    ).result_rows
    cols = client.query(
        'SELECT table, name, type, position FROM system.columns WHERE database = {d:String}', parameters=d
    ).result_rows
    names = client.query('SELECT name FROM system.dictionaries WHERE database = {d:String}', parameters=d).result_rows
    for (name,) in names:
        client.command(f'SYSTEM RELOAD DICTIONARY `{client.database}`.`{name}`')
    dictionaries = client.query(
        'SELECT name, key.names, key.types, attribute.names, attribute.types, type, lifetime_min, lifetime_max '
        'FROM system.dictionaries WHERE database = {d:String}',
        parameters=d,
    ).result_rows
    rows = {}
    for name, engine, *_ in tables:
        if engine != 'View':
            rows[name] = sorted(client.raw_query(f'SELECT * FROM `{name}` FORMAT TSV').decode().splitlines())  # noqa: S608 system names
    for name, *_ in dictionaries:
        query = f"SELECT * FROM dictionary('{client.database}.{name}') FORMAT TSV"  # noqa: S608 system names
        rows[f'dictionary {name}'] = sorted(client.raw_query(query).decode().splitlines())
    return {
        'tables': {name: tuple(rest) for name, *rest in tables},
        'columns': {(t, n): (ty, p) for t, n, ty, p in cols},
        'dictionaries': {name: tuple(rest) for name, *rest in dictionaries},
        'rows': rows,
    }


def _accepted(old: dict) -> dict:
    """The script tables `old` with the accepted engine differences applied, each checked to be there.

    A table is (engine, engine_full, sorting_key, primary_key).
    """
    accepted = dict(old)
    for name in PRIMARY_KEY_IS_SORT_KEY & old.keys():
        engine, full, sorting, _ = old[name]
        assert ' PRIMARY KEY ' in full, f'{name}: the script declared no primary key'
        accepted[name] = (engine, re.sub(r' PRIMARY KEY .*? ORDER BY ', ' ORDER BY ', full), sorting, sorting)
    for name in NULLABLE_KEY_DROPPED & old.keys():
        engine, full, *keys = accepted[name]
        assert 'allow_nullable_key = 1, ' in full, f'{name}: the script set no allow_nullable_key'
        accepted[name] = (engine, full.replace('allow_nullable_key = 1, ', ''), *keys)
    return accepted


def _without(shape: dict, names: set[str]) -> dict:
    return {
        'tables': {n: v for n, v in shape['tables'].items() if n not in names},
        'columns': {k: v for k, v in shape['columns'].items() if k[0] not in names},
        'dictionaries': shape['dictionaries'],
        'rows': {n: v for n, v in shape['rows'].items() if n not in names},
    }


@pytest.mark.parametrize('product', ['platform', 'ppp'])
def test_declared_tables_match_scripts(product):
    """From the same landing rows, the declarations build what the scripts they replace built: tables, sort keys,
    columns, dictionaries and rows; engines and engine settings but for the accepted differences."""
    loaded = {task['dataset'] for task in _tasks(product) if 'dataset' in task}
    assert set(DECLARED) - loaded == NOT_LOADED[product]
    datasets = [d for d in DECLARED if d in loaded]
    with _database() as before, _database() as after:
        for dataset in datasets:
            entry = DATASETS[dataset]
            for read in SCRIPT_READS.get(dataset, []):
                _land(before, read, product, script=True)
                for sql in split_statements((BEFORE / f'{read}.sql').read_text()):
                    before.command(sql)
            for read in SCRIPT_LANDINGS.get(dataset, []):
                _land(before, read, product)
            _land(before, dataset, product, script=True)
            for sql in split_statements((BEFORE / f'{dataset}.sql').read_text()):
                before.command(sql)
            for read in SCRIPT_READS.get(dataset, []):
                before.command(f'DROP TABLE `{read}` SYNC')
            for read in SCRIPT_LANDINGS.get(dataset, []):
                before.command(f'DROP TABLE `{DATASETS[read]["table"]}` SYNC')
            _land(after, dataset, product)
            for table in declarations(entry, product)[0]:
                if table.engine == 'rocksdb':
                    after.command(uniqueness_sql(table, entry_key(entry, product), entry['table']))
            for sql in declared_sql(entry, product):
                after.command(sql)
        relations = before.query('SELECT DISTINCT synonyms.relation FROM disease').result_rows
        old, new = _shape(before), _shape(after)
    # pts writes one entry per Synonyms field in field order, the order the script's tupleToNameValuePairs gives
    assert relations == [(list(Synonyms.model_fields),)]
    assert set(old['tables']) - NOT_REBUILT == set(new['tables'])
    for name in RESORTED:
        assert old['tables'][name][0] == new['tables'][name][0] == 'MergeTree', name
        columns_of = [{k: v for k, v in shape['columns'].items() if k[0] == name} for shape in (old, new)]
        assert columns_of[0] == columns_of[1], name
        if name in CONTENT_FROM_PTS:
            assert len(old['rows'][name]) == len(new['rows'][name]), name
        else:
            assert old['rows'][name] == new['rows'][name], name
    old, new = _without(old, NOT_REBUILT | RESORTED), _without(new, RESORTED)
    assert new['tables'] == _accepted(old['tables'])
    assert new['columns'] == old['columns']
    assert new['dictionaries'] == old['dictionaries']
    assert new['rows'].keys() == old['rows'].keys()
    for name, rows in old['rows'].items():
        assert rows, f'{name}: no fixture rows reached it'
        if name in CONTENT_FROM_PTS:
            assert len(new['rows'][name]) == len(rows), name
        else:
            assert new['rows'][name] == rows, name
    assert 0 < len(new['rows']['evidence_by_variant']) < len(new['rows']['evidence']), 'the filter dropped nothing'
    # the other filters cannot drop a row: their columns are not Nullable
    for table, column in NON_NULL_FILTERS:
        assert not new['columns'][table, column][0].startswith('Nullable'), (table, column)


#: Columns a declaration serves under another name than the script it replaces did, by table.
RENAMED = {'pharmacogenomics': {'rowId': 'pharmacogenomicsId'}}


def _renamed(old: dict) -> dict:
    """The script shape `old` with the `RENAMED` columns renamed, each checked to be there."""
    tables, cols = dict(old['tables']), dict(old['columns'])
    for table, renames in RENAMED.items():
        for before, after in renames.items():
            assert (table, before) in cols, f'{table}: the script served no {before}'
            cols[table, after] = cols.pop((table, before))
            tables[table] = tuple(re.sub(rf'\b{before}\b', after, v) for v in tables[table])
    return {**old, 'tables': tables, 'columns': cols}


def _build_declared(client, dataset: str, product: str) -> None:
    """The dataset's declared tables from its filled landing table, as the loader builds them."""
    entry = DATASETS[dataset]
    for table in declarations(entry, product)[0]:
        if table.engine == 'rocksdb':
            ClickhouseLoad._check_unique(client, table, entry_key(entry, product), entry['table'])
    for sql in declared_sql(entry, product):
        client.command(sql)


def test_pharmacogenomics_declaration_serves_what_its_script_served():
    """From the same landing rows, the declaration serves the table the script it replaces served: engine, key,
    columns and rows, but for the key `rowId` renamed `pharmacogenomicsId`."""
    with _database() as before, _database() as after:
        for client in (before, after):
            _land(client, 'pharmacogenomics', 'platform')
        _run(before, (BEFORE / 'pharmacogenomics_by_id.sql').read_text())
        _build_declared(after, 'pharmacogenomics', 'platform')
        old, new = _shape(before), _shape(after)
    assert set(new['tables']) == {'pharmacogenomics'}
    assert new['columns']['pharmacogenomics', 'pharmacogenomicsId'] == ('UInt64', 1)
    assert new['tables']['pharmacogenomics'][3] == 'pharmacogenomicsId'
    assert len(new['rows']['pharmacogenomics']) == FIXTURE_ROWS
    assert new == _renamed(old)


def _run(client, script: str) -> None:
    for statement in split_statements(script):
        client.command(statement)


def test_pharmacogenomics_load_rejects_repeated_ids(tmp_path):
    """A landing table with a repeated pharmacogenomicsId, or one loaded from a parquet without the column
    (ClickHouse fills 0), fails the load instead of collapsing the served table."""
    with _database() as repeated, _database() as missing:
        _land(repeated, 'pharmacogenomics', 'platform')
        repeated.command('INSERT INTO pharmacogenomics_log SELECT * FROM pharmacogenomics_log LIMIT 1')
        _land(missing, 'pharmacogenomics', 'platform')
        frame = pl.read_parquet(
            BytesIO(missing.raw_query('SELECT * EXCEPT (pharmacogenomicsId) FROM pharmacogenomics_log FORMAT Parquet'))
        )
        assert 'pharmacogenomicsId' not in frame.columns
        missing.command('TRUNCATE TABLE pharmacogenomics_log')
        file = tmp_path / 'old.parquet'
        frame.write_parquet(file)
        insert_file(missing, 'pharmacogenomics_log', str(file), fmt='Parquet', settings=INSERT_SETTINGS)
        for client in (repeated, missing):
            with pytest.raises(ClickhouseLoadError, match='duplicate pharmacogenomicsId in pharmacogenomics'):
                _build_declared(client, 'pharmacogenomics', 'platform')
            assert not client.command('EXISTS TABLE pharmacogenomics')


#: Fixture disease ids, and the fixture associations naming one: every other row.
DISEASE_ID = "concat('D', toString(rowNumberInAllBlocks()))"
NAMES_A_DISEASE = (
    f"if(rowNumberInAllBlocks() % 2 = 0, concat('D', toString(rowNumberInAllBlocks() % {FIXTURE_ROWS})), diseaseId)"
)
MEASUREMENT = "if(rowNumberInAllBlocks() % 3 = 0, arrayPushBack(therapeuticAreas, 'EFO_0001444'), therapeuticAreas)"
#: The aotf declarations, which replace the frozen aotf.sql.
AOTF = ('aotf_disease', 'aotf_target')
#: The landing columns aotf.sql read (the removed `association_otf` view), by the disease-side column each became.
AOTF_LANDING = {
    'rowId': 'rowId', 'diseaseId': 'A', 'targetId': 'B', 'datasourceId': 'datasourceId', 'datatypeId': 'datatypeId',
    'rowScore': 'rowScore', 'diseaseData': 'searchA', 'targetData': 'searchB', 'noveltyDirect': 'noveltyDirect',
    'noveltyIndirect': 'noveltyIndirect',
}  # fmt: skip
#: The served column the declarations retype: (before, after).
AOTF_RETYPED = {('associations_otf_target', 'isMeasurement'): ('UInt8', 'Bool')}


def test_aotf_loads_from_its_declarations():
    """Both aotf entries load in the declared explode; the scripted `aotf` load is gone."""
    (explode,) = [t for t in STEP if t['name'] == 'explode load declared datasets']
    assert set(AOTF) <= set(explode['foreach'])
    assert all(is_declared(DATASETS[d]) for d in AOTF)
    assert 'aotf' not in DATASETS
    assert not [t['name'] for t in STEP if t.get('dataset') == 'aotf']


@pytest.mark.parametrize('product', ['platform', 'ppp'])
def test_aotf_declarations_build_what_aotf_sql_built(product):
    """The aotf declarations build the `associations_otf_*` tables the frozen aotf.sql built over the served
    `disease`: tables, engines, sort keys and columns, but for target `isMeasurement`, UInt8 there and Bool here.
    pts computes their rows (tests/test_association_otf_polars.py), so only the counts are compared."""
    served = {'associations_otf_disease', 'associations_otf_target'}
    disease = DATASETS['disease']
    landed = dict(landing_columns(DATASETS['aotf_disease'], product) or [])
    landing = [(old, landed[new]) for old, new in AOTF_LANDING.items()]
    with _database() as before, _database() as after:
        for _, sql in ddl_for(disease, product):
            before.command(sql)
        overrides = {'id': DISEASE_ID, 'therapeuticAreas': MEASUREMENT}
        _fill(before, disease['table'], landing_columns(disease, product), 1, overrides)
        _build_declared(before, 'disease', product)
        before.command(create_table('associations_otf_log', landing))
        _fill(before, 'associations_otf_log', landing, 2, {'diseaseId': NAMES_A_DISEASE})
        _run(before, (BEFORE / 'aotf.sql').read_text())
        joined = before.query(
            'SELECT countIf(isMeasurement), countIf(indirect != []) FROM associations_otf_target'
        ).result_rows[0]
        for dataset in AOTF:
            _land(after, dataset, product)
            _build_declared(after, dataset, product)
        old, new = _shape(before), _shape(after)
    assert all(joined), f'the join matched no disease: {joined}'
    old = _without(old, {'disease', 'disease_mergetree', 'disease_names', 'dictionary disease_names'})
    assert set(old['tables']) == set(new['tables']) == served
    assert new['tables'] == old['tables']
    for key, (was, now) in AOTF_RETYPED.items():
        assert old['columns'][key][0] == was, key
        old['columns'][key] = (now, old['columns'][key][1])
    assert new['columns'] == old['columns']
    counts = {name: len(rows) for name, rows in new['rows'].items()}
    assert counts == {name: len(rows) for name, rows in old['rows'].items()} == dict.fromkeys(served, FIXTURE_ROWS)


#: Fixture rows: interactions with unique targetA and interactionId, each named by one evidence.
UNIQUE_TARGET = "concat('T', toString(rowNumberInAllBlocks()))"
EVIDENCE_OF = {'interactionId': f'rowNumberInAllBlocks() % {FIXTURE_ROWS}'}


def test_interaction_loads_from_its_declaration():
    """`interaction` loads in the declared explode; `interaction_evidence` stays an undeclared landing table with no
    script, which the API reads."""
    (explode,) = [t for t in STEP if t['name'] == 'explode load declared datasets']
    assert 'interaction' in explode['foreach']
    assert is_declared(DATASETS['interaction'])
    assert not [t['name'] for t in STEP if t.get('dataset') == 'interaction']
    evidence = DATASETS['interaction_evidence']
    assert not is_declared(evidence)
    assert 'postload_script' not in evidence


@pytest.mark.parametrize('product', ['platform', 'ppp'])
def test_interaction_declaration_builds_what_interaction_sql_built(product):
    """The declaration builds the `interaction` table the frozen interaction.sql built from `interaction_log` and
    `interaction_evidence_log`: engine, key and columns. pts computes its rows (tests/test_interaction_view.py), so
    only the counts are compared. `interaction_log` is no longer left behind; `interaction_evidence_log` stays."""
    evidence = DATASETS['interaction_evidence']
    script_landing = {'table': 'interaction_log', 'model': 'interaction'}
    with _database() as before, _database() as after:
        for client in (before, after):
            for _, sql in ddl_for(evidence, product):
                client.command(sql)
            _fill(client, evidence['table'], landing_columns(evidence, product), 2, EVIDENCE_OF)
        for _, sql in ddl_for(script_landing, product):
            before.command(sql)
        overrides = {'targetA': UNIQUE_TARGET, 'interactionId': 'rowNumberInAllBlocks()'}
        _fill(before, 'interaction_log', landing_columns(script_landing, product), 1, overrides)
        _run(before, (BEFORE / 'interaction.sql').read_text())
        entry = DATASETS['interaction']
        for _, sql in ddl_for(entry, product):
            after.command(sql)
        _fill(after, entry['table'], landing_columns(entry, product), 1, {'targetA': UNIQUE_TARGET})
        _build_declared(after, 'interaction', product)
        empty = before.query('SELECT countIf(empty(i.evidences)) FROM interaction ARRAY JOIN interactions AS i')
        old, new = _shape(before), _shape(after)
    assert empty.result_rows == [(0,)], 'an interaction got no evidence'
    old = _without(old, {'interaction_log'})
    assert set(new['tables']) == {'interaction', 'interaction_evidence_log'}
    assert new['tables'] == old['tables']
    assert new['columns'] == old['columns']
    assert new['rows']['interaction_evidence_log'] == old['rows']['interaction_evidence_log']
    assert len(new['rows']['interaction']) == len(old['rows']['interaction']) == FIXTURE_ROWS
