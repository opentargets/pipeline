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

from pos.clickhouse_ddl import columns, landing_ddl, landing_table, render, served_sql, served_views
from pos.services.clickhouse import connect
from pos.tasks.clickhouse_load import INSERT_SETTINGS, ClickhouseLoadError, build_served
from pos.tasks.clickhouse_load_views import view_loads

ROOT = Path(__file__).parents[1]
STEP = yaml.safe_load((ROOT / 'config' / 'config.yaml').read_text())['steps']['backend_clickhouse']
VIEWS = served_views()
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
    """The step's tasks for the product, served-view children expanded, in `requires:` order; requiring the
    generating task requires its children."""
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


def _statements(dataset: str) -> list[tuple[str, str]]:
    """(what, statement) a load of the view runs, tables empty."""
    model = VIEWS[dataset]
    return [('generated DDL', landing_ddl(model)), *(('served tables', sql) for sql in served_sql(model))]


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
            for what, sql in _statements(task['dataset']):
                try:
                    client.command(sql)
                except ClickHouseError as e:
                    pytest.fail(f'{task["name"]}: {what} failed: {e}\nstatement: {sql}')
        names = client.query(
            'SELECT name FROM system.tables WHERE database = {d:String}', parameters={'d': client.database}
        )
        left = {name for (name,) in names.result_rows if name.endswith('_log')}
        assert not left, f'landing tables left after the load: {left}'


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
#: of structs; disease_phenotype_by_disease_view: Nullable, in a list of structs in a list of structs)).
ENUM_MODELS = (
    'clinical_report_view',
    'colocalisation_by_credible_set_view',
    'variant_view',
    'interaction_view',
    'evidence_gene_burden_view',
    'target_by_region_view',
    'target_view',
    'drug_mechanism_of_action_by_drug_view',
    'disease_phenotype_by_disease_view',
    'transcript_view',
)


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
    table, create = landing_table(model), landing_ddl(model)
    with _database() as client:
        client.command(create)
        insert_file(client, table, str(file), fmt='Parquet', settings=INSERT_SETTINGS)
        raw = client.raw_query('SELECT * FROM {t:Identifier} FORMAT JSONEachRow', parameters={'t': table}).decode()
    back = [json.loads(line) for line in raw.splitlines()]
    assert len(back) == len(rows)
    for want, got in zip(frame.to_dicts(), back, strict=True):
        for name, dtype in schema.items():
            assert list(_labels(got[name], dtype, name)) == list(_labels(want[name], dtype, name)), name


#: The task that loads every served view but colocalisation.
SERVED_LOADS = 'clickhouse_load_views load served views'
COLOCALISATION = 'colocalisation_by_credible_set_view'
#: Entries and tasks the served views replaced.
RETIRED = (
    'aotf', 'credible_set_variant', 'credible_sets_build_cleanup', 'protein_coding_coords',
    # served through their views
    'baseline_expression', 'biosample', 'clinical_report', 'enhancer_to_gene', 'go', 'hpo', 'interaction_evidence',
    'literature_entity_lut', 'otar_projects', 'pharmacogenomics', 'so', 'target_essentiality', 'w2v',
    'evidence', 'evidence_by_variant', 'evidence_by_disease_and_target', 'clinical_indication', 'clinical_target',
    'clinical_indication_drug', 'clinical_indication_disease',
)  # fmt: skip


def test_the_served_views_load_in_their_task_and_colocalisation_after():
    """Every load is a served view's: the served views load in their own task, and colocalisation loads alone after
    it, its RocksDB insert being the step's memory peak."""
    (served,) = [t for t in STEP if t['name'] == SERVED_LOADS]
    assert not [t['name'] for t in STEP if 'foreach' in t]
    assert served['exclude'] == [COLOCALISATION]
    loaded = [t['dataset'] for t in STEP if 'dataset' in t and t['name'].startswith('clickhouse_load')]
    assert loaded == [COLOCALISATION]
    (coloc,) = [t for t in STEP if t.get('dataset') == COLOCALISATION]
    assert coloc['requires'] == [served['name']]
    assert 'products' not in coloc
    assert set(RETIRED).isdisjoint(VIEWS)
    assert not [name for name in VIEWS if name.endswith('_mergetree')]


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


def _land(client, dataset: str) -> None:
    """The view's landing table, filled with fixture rows."""
    model = VIEWS[dataset]
    client.command(landing_ddl(model))
    _fill(client, landing_table(model), columns(model), 1)


def test_pharmacogenomics_load_rejects_repeated_ids(tmp_path):
    """A landing table with a repeated pharmacogenomicsId, or one loaded from a parquet without the column
    (ClickHouse fills 0), fails the load instead of collapsing the served table."""
    with _database() as repeated, _database() as missing:
        _land(repeated, 'pharmacogenomics_view')
        repeated.command('INSERT INTO pharmacogenomics_view_log SELECT * FROM pharmacogenomics_view_log LIMIT 1')
        _land(missing, 'pharmacogenomics_view')
        sql = 'SELECT * EXCEPT (pharmacogenomicsId) FROM pharmacogenomics_view_log FORMAT Parquet'
        frame = pl.read_parquet(BytesIO(missing.raw_query(sql)))
        assert 'pharmacogenomicsId' not in frame.columns
        missing.command('TRUNCATE TABLE pharmacogenomics_view_log')
        file = tmp_path / 'old.parquet'
        frame.write_parquet(file)
        insert_file(missing, 'pharmacogenomics_view_log', str(file), fmt='Parquet', settings=INSERT_SETTINGS)
        for client in (repeated, missing):
            with pytest.raises(ClickhouseLoadError, match='duplicate pharmacogenomicsId in pharmacogenomics_view'):
                build_served(client, VIEWS['pharmacogenomics_view'])
            assert not client.command('EXISTS TABLE pharmacogenomics_view')
