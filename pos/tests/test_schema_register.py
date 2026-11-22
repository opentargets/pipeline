"""docs/phase5_register.csv: the phase 5 change register covers every model/DDL disagreement."""

import csv
import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip('ot_datamodels')

import yaml
from ot_datamodels.discovery import discover_datasets
from ot_datamodels.spec import dataset_spec, walk

ROOT = Path(__file__).parents[1]
CLICKHOUSE = ROOT / 'config' / 'clickhouse'
REGISTER = ROOT / 'docs' / 'phase5_register.csv'
COLUMNS = ['dataset', 'field', 'category', 'outcome', 'before', 'after', 'evidence', 'serving read', 'API follow-up']
#: Outcomes that are a model change, so their disagreement leaves drift().
MODEL_CHANGES = {'marker', 'model nullable'}
#: Categories whose serving read is the one the column would have once loaded.
UNLANDED = {'model_only', 'struct_field_missing'}
#: Categories whose `before` is not a landing DDL cell.
NOT_LANDED = {*UNLANDED, 'no_model'}
ORDER = {'struct_order', 'column_order'}


def _load_script():
    spec = importlib.util.spec_from_file_location('schema_drift', ROOT / 'scripts' / 'schema_drift.py')
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules['schema_drift'] = module
    spec.loader.exec_module(module)
    return module


schema_drift = _load_script()


@pytest.fixture(scope='module')
def drift() -> dict[tuple[str, str, str], dict[str, str]]:
    """(dataset, field, category) -> the drift row; evidence models share the DDL cells, the first is kept."""
    frame = schema_drift.drift(ROOT / 'config' / 'datasets.yaml', CLICKHOUSE / 'schema', CLICKHOUSE / 'scripts')
    rows: dict[tuple[str, str, str], dict[str, str]] = {}
    for r in frame.iter_rows(named=True):
        rows.setdefault((r['dataset'], r['field'], r['category']), r)
    return rows


@pytest.fixture(scope='module')
def register() -> list[dict[str, str]]:
    with REGISTER.open(newline='') as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames == COLUMNS
        return list(reader)


def _nodes(dataset: str) -> list[dict]:
    """The `walk` nodes of each model the landing table is loaded from."""
    entry = yaml.safe_load((ROOT / 'config' / 'datasets.yaml').read_text())['clickhouse'][dataset]
    models = schema_drift._models_for(entry, discover_datasets()) or []
    return [dict(walk(dataset_spec(m))) for m in models]


def _changed(row: dict[str, str]) -> bool:
    """Whether the ClickHouse type (or the struct field order) differs between `before` and `after`."""
    before, after = row['before'], row['after']
    if row['category'] in ORDER or not before or not after:
        return before != after
    return schema_drift.parse_type(before) != schema_drift.parse_type(after)


def _in_nullable_struct(dataset: str, path: str) -> bool:
    """Whether an ancestor of `path` in a model of the dataset is a nullable struct."""
    parts = path.split('/')
    ancestors = ['/'.join(parts[:i]) for i in range(1, len(parts))]
    return any(n[a].kind == 'struct' and n[a].nullable for n in _nodes(dataset) for a in ancestors if a in n)


def _check_against_drift(row: dict[str, str], found: dict[str, str]) -> None:
    key = (row['dataset'], row['field'], row['category'])
    if row['category'] not in NOT_LANDED:
        assert row['before'] == found['ddl_type'], f'{key}: before should be {found["ddl_type"]}'
    if row['category'] not in UNLANDED:
        assert row['serving read'] == found['serving_read'], f'{key}: serving read is {found["serving_read"]}'
    checkable = row['after'] and row['category'] not in ORDER and row['dataset'] != 'evidence'
    if checkable and not _in_nullable_struct(row['dataset'], row['field']):
        after = schema_drift.parse_type(row['after'])
        if after.name not in {'Array', 'Tuple'}:
            nullable = found['model_nullable'] == 'true'
            assert after.nullable == nullable, f'{key}: after should be {"" if nullable else "non-"}nullable'


def test_register_covers_every_disagreement(drift, register):
    keys = [(r['dataset'], r['field'], r['category']) for r in register]
    assert len(keys) == len(set(keys)), 'duplicate register rows'
    missing = sorted(set(drift) - set(keys))
    assert not missing, f'disagreements with no register row: {missing}'

    for row, key in zip(register, keys, strict=True):
        served = row['serving read'] != 'no' and row['category'] != 'column_order'
        expected = 'open' if served and _changed(row) else 'none'
        assert row['API follow-up'] == expected, f'{key}: API follow-up should be {expected}'

        if key in drift:
            _check_against_drift(row, drift[key])
            continue
        # no longer a disagreement: only a model change may have removed it
        assert row['outcome'] in MODEL_CHANGES, f'{key}: stale row, not in drift() and not a model change'
        nodes = [n[row['field']] for n in _nodes(row['dataset']) if row['field'] in n]
        assert nodes, f'{key}: no model of {row["dataset"]} has {row["field"]}'
        if row['outcome'] == 'marker':
            marked = [replace(schema_drift.parse_type(n.clickhouse), nullable=False) for n in nodes if n.clickhouse]
            after = replace(schema_drift.parse_type(row['after']), nullable=False)
            assert after in marked, f'{key}: no such marker'
        else:
            assert any(n.nullable for n in nodes), f'{key}: not nullable in the model'
