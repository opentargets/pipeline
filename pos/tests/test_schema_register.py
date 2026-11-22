"""docs/phase5_register.csv: the phase 5 change register covers every model/DDL disagreement."""

import csv
import importlib.util
import sys
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


def _load_script():
    spec = importlib.util.spec_from_file_location('schema_drift', ROOT / 'scripts' / 'schema_drift.py')
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules['schema_drift'] = module
    spec.loader.exec_module(module)
    return module


schema_drift = _load_script()


@pytest.fixture(scope='module')
def drift() -> dict[tuple[str, str, str], str]:
    """(dataset, field, category) -> serving_read, evidence models merged."""
    frame = schema_drift.drift(ROOT / 'config' / 'datasets.yaml', CLICKHOUSE / 'schema', CLICKHOUSE / 'scripts')
    return {(r['dataset'], r['field'], r['category']): r['serving_read'] for r in frame.iter_rows(named=True)}


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


def test_register_covers_every_disagreement(drift, register):
    keys = [(r['dataset'], r['field'], r['category']) for r in register]
    assert len(keys) == len(set(keys)), 'duplicate register rows'
    missing = sorted(set(drift) - set(keys))
    assert not missing, f'disagreements with no register row: {missing}'

    for row, key in zip(register, keys, strict=True):
        changed = row['before'] != row['after'] and row['category'] != 'column_order'
        served = row['serving read'] != 'no'
        expected = 'open' if changed and served else 'none'
        assert row['API follow-up'] == expected, f'{key}: API follow-up should be {expected}'

        if key in drift:
            if row['category'] not in UNLANDED:
                assert row['serving read'] == drift[key], f'{key}: serving read is {drift[key]}'
            continue
        # no longer a disagreement: only a model change may have removed it
        assert row['outcome'] in MODEL_CHANGES, f'{key}: stale row, not in drift() and not a model change'
        nodes = [n[row['field']] for n in _nodes(row['dataset']) if row['field'] in n]
        assert nodes, f'{key}: no model of {row["dataset"]} has {row["field"]}'
        if row['outcome'] == 'marker':
            assert any(n.clickhouse and n.clickhouse in row['after'] for n in nodes), f'{key}: no marker'
        else:
            assert any(n.nullable for n in nodes), f'{key}: not nullable in the model'
