"""The step's ClickHouse SQL runs against the generated landing tables, on a live ClickHouse.

`POS_TEST_CLICKHOUSE=<host>:<port>` (default `localhost:8123`) names the server; with no server the test skips,
unless `POS_REQUIRE_CLICKHOUSE=1`.
"""

import datetime
import json
import os
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
from ot_datamodels.polars import to_polars_schema

from pos.clickhouse_ddl import (
    clickhouse_entries,
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
    render,
    served_entries,
)
from pos.clickhouse_tables import entry_tables
from pos.services.clickhouse import connect
from pos.tasks.clickhouse_load import INSERT_SETTINGS, ClickhouseLoad, ClickhouseLoadError, member_drops, union_inserts
from pos.tasks.clickhouse_load_views import view_loads

ROOT = Path(__file__).parents[1]
STEP = yaml.safe_load((ROOT / 'config' / 'config.yaml').read_text())['steps']['backend_clickhouse']
DATASETS = clickhouse_entries(ROOT / 'config' / 'datasets.yaml')
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
    """The step's tasks for the product, explode and served-view children expanded, in `requires:` order;
    requiring a generating task requires its children."""
    by_name: dict[str, dict] = {}
    children: dict[str, list[str]] = {}
    for task in STEP:
        if product not in task.get('products', [product]):
            continue
        if task['name'].startswith('clickhouse_load_views '):
            children[task['name']] = []
            for view in view_loads(product, task.get('exclude', [])):
                name = f'clickhouse_load clickhouse load {view}'
                by_name[name] = {'name': name, 'dataset': view, 'requires': task.get('requires', [])}
                children[task['name']].append(name)
        elif 'foreach' in task:
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
    """(what, statement) a load of the dataset runs, tables empty."""
    entry = DATASETS[dataset]
    out = [('generated DDL', sql) for _, sql in ddl_for(entry, product)]
    if 'models' in entry:
        loaded = members(entry['models'], product)
        out += [('member inserts', sql) for sql in union_inserts(entry['table'], loaded)]
        out += [('member drops', sql) for sql in member_drops(loaded)]
    return out + [('declared tables', sql) for sql in declared_sql(entry, product)]


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
            for what, sql in _statements(task['dataset'], product):
                try:
                    client.command(sql)
                except ClickHouseError as e:
                    pytest.fail(f'{task["name"]}: {what} failed: {e}\nstatement: {sql}')
        names = client.query(
            'SELECT name FROM system.tables WHERE database = {d:String}', parameters={'d': client.database}
        )
        left = {name for (name,) in names.result_rows if name.endswith('_log')}
        assert left == {'interaction_evidence_log'}, f'landing tables left after the load: {left}'


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
#: interaction_view: in a list of structs; drug_mechanism_of_action_by_drug_view: a top-level list, and in a list
#: of structs; disease_phenotype_by_disease_view: Nullable, in a list of structs in a list of structs), and the
#: dataset each loads into.
ENUM_MODELS = {
    'clinical_report': 'clinical_report',
    'colocalisation_by_credible_set_view': 'colocalisation_by_credible_set_view',
    'variant_view': 'variant_view',
    'interaction_view': 'interaction_view',
    'evidence_gene_burden': 'evidence',
    'target_by_region_view': 'target_by_region_view',
    'target_view': 'target_view',
    'drug_mechanism_of_action_by_drug_view': 'drug_mechanism_of_action_by_drug_view',
    'disease_phenotype_by_disease_view': 'disease_phenotype_by_disease_view',
    'transcript_view': 'transcript_view',
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


#: The explode every declared datasets.yaml entry but the PPP-only `otar_projects` loads in.
DECLARED_LOADS = 'explode load declared datasets'
#: The task that loads every served view but colocalisation.
SERVED_LOADS = 'clickhouse_load_views load served views'
COLOCALISATION = 'colocalisation_by_credible_set_view'
#: Entries and tasks the declarations replaced.
RETIRED = ('aotf', 'credible_set_variant', 'credible_sets_build_cleanup', 'protein_coding_coords')


def test_declared_datasets_load_in_the_declared_explode():
    """The declared explode is the step's only one and the served views load in their own task; colocalisation
    loads alone after both, its RocksDB insert being the step's memory peak; every other load waits only on what
    they wait on; `interaction_evidence` is an undeclared landing table, which the API reads."""
    (explode,) = [t for t in STEP if t['name'] == DECLARED_LOADS]
    (served,) = [t for t in STEP if t['name'] == SERVED_LOADS]
    assert [t['name'] for t in STEP if 'foreach' in t] == [explode['name']]
    assert all(is_declared(DATASETS[d]) for d in explode['foreach'])
    assert set(explode['foreach']).isdisjoint(served_entries())
    assert served['exclude'] == [COLOCALISATION]
    assert served['requires'] == explode['requires']
    loaded = [t['dataset'] for t in STEP if 'dataset' in t and t['name'].startswith('clickhouse_load')]
    assert set(loaded).isdisjoint(explode['foreach'])
    assert set(loaded) & set(served_entries()) == {COLOCALISATION}
    (coloc,) = [t for t in STEP if t.get('dataset') == COLOCALISATION]
    assert coloc['requires'] == [explode['name'], served['name']]
    assert 'products' not in coloc
    assert 'interaction_evidence' in loaded
    assert not is_declared(DATASETS['interaction_evidence'])
    assert set(RETIRED).isdisjoint(DATASETS)
    assert set(RETIRED).isdisjoint(loaded)
    for name, entry in served_entries().items():
        assert entry['input_dir'] == f'view/{name}', name
    assert not [t.name for e in DATASETS.values() for t in entry_tables(e) if t.name.endswith('_mergetree')]
    loads = [t for t in STEP if t['name'].startswith('clickhouse_load') or 'foreach' in t]
    allowed = {*explode['requires'], explode['name'], served['name']}
    assert all(set(t.get('requires', [])) <= allowed for t in loads)


#: Fixture rows per landing table; generateRandom makes a few percent of a Nullable column NULL.
FIXTURE_ROWS = 300
#: Columns made unique in the fixture rows.
KEYS = {'pharmacogenomicsId'}


def _unique(name: str, dtype, seed: int) -> str:
    if render(dtype).startswith('UInt'):
        return f'rowNumberInAllBlocks() AS `{name}`'
    return f"concat('{seed}-', toString(rowNumberInAllBlocks()), `{name}`) AS `{name}`"


def _fill(client, table: str, cols, seed: int) -> None:
    """`FIXTURE_ROWS` rows into `table`, the same for the same `seed` on every database."""
    structure = ', '.join(f'`{name}` {render(t)}' for name, t in cols)
    literal = structure.replace('\\', '\\\\').replace("'", "\\'")
    unique = [_unique(n, t, seed) for n, t in cols if n in KEYS]
    replace = f' REPLACE ({", ".join(unique)})' if unique else ''
    client.command(
        f"INSERT INTO `{table}` SELECT *{replace} FROM generateRandom('{literal}', {seed}, 12, 3) "  # noqa: S608 generated names
        f'LIMIT {FIXTURE_ROWS}'
    )


def _land(client, dataset: str, product: str) -> None:
    """The dataset's landing table, filled with fixture rows."""
    entry = DATASETS[dataset]
    for _, sql in ddl_for(entry, product):
        client.command(sql)
    _fill(client, entry['table'], landing_columns(entry, product), 1)


def _build_declared(client, dataset: str, product: str) -> None:
    """The dataset's declared tables from its filled landing table, as the loader builds them."""
    entry = DATASETS[dataset]
    for table in declarations(entry, product)[0]:
        if table.engine == 'rocksdb':
            ClickhouseLoad._check_unique(client, table, entry_key(entry, product), entry['table'])
    for sql in declared_sql(entry, product):
        client.command(sql)


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
