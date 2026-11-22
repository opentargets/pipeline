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

from pos.clickhouse_ddl import CHType, columns, members, parse_type, render, union_columns

ROOT = Path(__file__).parents[1]
CLICKHOUSE = ROOT / 'config' / 'clickhouse'
REGISTER = ROOT / 'docs' / 'phase5_register.csv'
SNAPSHOT = ROOT / 'docs' / 'phase5_drift_before.csv'
COLUMNS = ['dataset', 'field', 'category', 'outcome', 'before', 'after', 'evidence', 'serving read', 'API follow-up']
ORDER = {'struct_order', 'column_order'}
#: Evidence columns only PPP models have.
PPP_ONLY = (
    'assays', 'assessments', 'biomarkerList', 'cellLineBackground', 'geneticInteractionScore',
    'geneticInteractionType', 'interactingTargetFromSourceId', 'interactingTargetRole',
    'phenotypicConsequenceLogFoldChange', 'phenotypicConsequencePValue', 'primaryProjectHit', 'primaryProjectId',
    'projectDescription', 'targetRole', 'validationReadouts',
)  # fmt: skip


def _load_script():
    spec = importlib.util.spec_from_file_location('schema_drift', ROOT / 'scripts' / 'schema_drift.py')
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules['schema_drift'] = module
    spec.loader.exec_module(module)
    return module


schema_drift = _load_script()


@pytest.fixture(scope='module')
def drift() -> set[tuple[str, str, str]]:
    """(dataset, field, category) of every disagreement drift() finds now."""
    frame = schema_drift.drift(ROOT / 'config' / 'datasets.yaml', CLICKHOUSE / 'schema', CLICKHOUSE / 'scripts')
    return {(r['dataset'], r['field'], r['category']) for r in frame.iter_rows(named=True)}


@pytest.fixture(scope='module')
def snapshot() -> set[tuple[str, str, str]]:
    """(dataset, field, category) of every disagreement before the piece 1 model changes (e9133802)."""
    with SNAPSHOT.open(newline='') as fh:
        return {(r['dataset'], r['field'], r['category']) for r in csv.DictReader(fh)}


@pytest.fixture(scope='module')
def register() -> list[dict[str, str]]:
    with REGISTER.open(newline='') as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames == COLUMNS
        return list(reader)


def _platform_table(dataset: str) -> CHType | None:
    """The generated Platform landing table of a dataset as one Tuple; None for a table with no model."""
    if dataset == 'evidence':
        return CHType('Tuple', fields=tuple(union_columns(members('evidence_*', 'platform'))))
    entry = yaml.safe_load((ROOT / 'config' / 'datasets.yaml').read_text())['clickhouse'][dataset]
    models = schema_drift._models_for(entry, discover_datasets())
    if not models:
        return None
    (model,) = models
    return CHType('Tuple', fields=tuple(columns(model)))


def _at(table: CHType | None, path: str) -> CHType | None:
    """The type at `a`, `a/b` or `a[]/b` (the table itself for ''); None if the path is absent."""
    t = table
    for part in path.split('/') if path else []:
        name, lists = part, 0
        while name.endswith('[]'):
            name, lists = name[:-2], lists + 1
        t = dict(t.fields).get(name) if t is not None else None
        for _ in range(lists):
            t = t.element if t is not None else None
    return t


def _changed(row: dict[str, str]) -> bool:
    """Whether the ClickHouse type (or the struct field order) differs between `before` and `after`."""
    before, after = row['before'], row['after']
    if row['category'] in ORDER or not before or not after:
        return before != after
    return parse_type(before) != parse_type(after)


def test_register_covers_every_disagreement(drift, snapshot, register):
    keys = [(r['dataset'], r['field'], r['category']) for r in register]
    assert len(keys) == len(set(keys)), 'duplicate register rows'
    unregistered = sorted(snapshot - set(keys))
    assert not unregistered, f'snapshot rows with no register row: {unregistered}'
    new = sorted(drift - snapshot - set(keys))
    assert not new, (
        f'new schema drift with no register row: {new}. Add a row by hand to pos/docs/phase5_register.csv '
        f'with columns {", ".join(COLUMNS)} (outcome says what changed; before/after are ClickHouse types; '
        f'evidence a short data summary; serving read from schema_drift.py; API follow-up open or none).'
    )
    for row, key in zip(register, keys, strict=True):
        served = row['serving read'] != 'no' and row['category'] != 'column_order'
        expected = 'open' if served and _changed(row) else 'none'
        assert row['API follow-up'] == expected, f'{key}: API follow-up should be {expected}'


def test_after_matches_the_generator(register):
    """`after` is the generated Platform type; empty where the generated table lacks the column."""
    tables = {d: _platform_table(d) for d in {r['dataset'] for r in register}}
    for row in register:
        key = (row['dataset'], row['field'], row['category'])
        found = _at(tables[row['dataset']], row['field'])
        if row['category'] in ORDER:
            kept = set(row['before'].split(', '))
            expected = ', '.join(n for n, _ in found.fields if n in kept) if found is not None else ''
        elif row['category'] in {'ddl_only', 'no_model'} or found is None:
            expected = ''
        else:
            expected = render(found, quote_fields=False)
        assert row['after'] == expected, f'{key}: after should be {expected!r}'


def test_ppp_only_columns_are_platform_removals(register):
    rows = {r['field']: r for r in register if r['dataset'] == 'evidence' and r['category'] == 'ppp_only'}
    assert sorted(rows) == sorted(PPP_ONLY)
    for field, row in rows.items():
        assert row['outcome'] == 'API removal (Platform; PPP only)', field
        assert row['after'] == '', field
        assert row['API follow-up'] == 'open', field
    platform = _platform_table('evidence')
    assert platform is not None
    assert not set(PPP_ONLY) & {n for n, _ in platform.fields}, 'a PPP-only column is in the Platform table'


def test_every_clickhouse_marker_parses():
    """Piece 2 generates DDL from the markers, so each one must parse."""
    for name, model in discover_datasets().items():
        for path, node in walk(dataset_spec(model)):
            if node.clickhouse:
                assert parse_type(node.clickhouse), f'{name}.{path}: {node.clickhouse}'
