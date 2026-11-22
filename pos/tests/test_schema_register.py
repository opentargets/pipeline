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

from pos.clickhouse_ddl import canonical, landing_columns, members, parse_type, render, type_at

ROOT = Path(__file__).parents[1]
CLICKHOUSE = ROOT / 'config' / 'clickhouse'
REGISTER = ROOT / 'docs' / 'phase5_register.csv'
SNAPSHOT = ROOT / 'docs' / 'phase5_drift_before.csv'
SQL_EDITS = ROOT / 'docs' / 'phase5_sql_edits.csv'
FOLLOW_UPS = ROOT / 'docs' / 'phase5_api_follow_ups.csv'
CAST_BACK = ROOT / 'docs' / 'phase5_cast_back.csv'
COLUMNS = ['dataset', 'field', 'category', 'outcome', 'before', 'after', 'evidence', 'serving read', 'API follow-up']
ORDER = {'struct_order', 'column_order'}
#: Evidence columns only PPP models have.
PPP_ONLY = (
    'assays', 'assessments', 'biomarkerList', 'cellLineBackground', 'geneticInteractionScore',
    'geneticInteractionType', 'interactingTargetFromSourceId', 'interactingTargetRole',
    'phenotypicConsequenceLogFoldChange', 'phenotypicConsequencePValue', 'primaryProjectHit', 'primaryProjectId',
    'projectDescription', 'targetRole', 'validationReadouts',
)  # fmt: skip


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / f'{name}.py')
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


schema_drift = _load_script('schema_drift')
build_register = _load_script('build_register')


@pytest.fixture(scope='module')
def drift() -> set[tuple[str, str, str]]:
    """(dataset, field, category) of every disagreement drift() finds now in the hand-written landing DDL.

    Generated entries have none by construction; the snapshot is the coverage list for their piece 1 rows.
    """
    frame = schema_drift.drift(ROOT / 'config' / 'datasets.yaml', CLICKHOUSE / 'schema', CLICKHOUSE / 'scripts')
    hand_written = {d for d, e in CONFIG.items() if 'preload_script' in e and not e.get('derived')}
    assert set(frame['dataset']) == hand_written
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


#: The header each reader of a committed `docs/*.csv` expects.
DOCS_HEADERS = {
    'phase5_api_follow_ups.csv': ['dataset', 'field', 'follow_up'],
    'phase5_cast_back.csv': ['dataset', 'field', 'table', 'path', 'served'],
    'phase5_drift_before.csv': schema_drift.COLUMNS,
    'phase5_landing_before.csv': ['dataset', 'field', 'ddl_type', 'position'],
    'phase5_register.csv': COLUMNS,
    'phase5_register_piece1.csv': COLUMNS,
    'phase5_sql_edits.csv': ['dataset', 'field', 'edit', 'table', 'served'],
}


@pytest.mark.parametrize('path', sorted((ROOT / 'docs').glob('*.csv')), ids=lambda p: p.name)
def test_docs_csv_parses(path):
    """Every committed `docs/*.csv` has the header its reader expects and that many cells on every row."""
    assert path.name in DOCS_HEADERS, f'{path.name}: no expected header in DOCS_HEADERS'
    with path.open(newline='') as fh:
        rows = list(csv.reader(fh))
    assert rows and rows[0] == list(DOCS_HEADERS[path.name]), f'{path.name}: header {rows[:1]}'
    short = [i for i, r in enumerate(rows[1:], 2) if len(r) != len(rows[0])]
    assert not short, f'{path.name}: lines {short} have the wrong number of cells'


CONFIG = yaml.safe_load((ROOT / 'config' / 'datasets.yaml').read_text())['clickhouse']
SUPERSEDED ='superseded by ppp_only'
SUPERSEDED_SQL = 'superseded by serving_sql'


def _platform(dataset: str):
    """The generated Platform landing columns of a dataset; None for a table with no model."""
    return landing_columns(CONFIG[dataset], 'platform')


def _changed(row: dict[str, str]) -> bool:
    """Whether the ClickHouse type (or the struct field order) differs between `before` and `after`."""
    before, after = row['before'], row['after']
    if row['category'] in ORDER or not before or not after:
        return before != after
    return not _same(before, after)


def _same(a: str, b: str) -> bool:
    return canonical(parse_type(a)) == canonical(parse_type(b))


def _cast_back() -> dict[tuple[str, str], dict[str, str]]:
    with CAST_BACK.open(newline='') as fh:
        return {(r['dataset'], r['field']): r for r in csv.DictReader(fh)}


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
    kept = {(r['dataset'], r['field']) for r in register if r['category'] == 'serving_sql' and not _changed(r)}
    cast_back = _cast_back()
    for row, key in zip(register, keys, strict=True):
        if row['category'] == 'api_follow_up':
            assert row['API follow-up'] == 'open', key
            continue
        served = row['serving read'] != 'no' and row['category'] != 'column_order'
        superseded = row['outcome'] in {SUPERSEDED, SUPERSEDED_SQL}
        if row['outcome'] == SUPERSEDED_SQL:
            assert key[:2] in kept, f'{key}: superseded by serving_sql, but no serving_sql row keeps the type'
        entry = cast_back.get(key[:2])
        if row['category'] == 'type_unrecorded' and entry is not None and _same(entry['served'], row['before']):
            superseded = True
        expected = 'open' if served and _changed(row) and not superseded else 'none'
        assert row['API follow-up'] == expected, f'{key}: API follow-up should be {expected}'


def test_after_matches_the_generator(register):
    """`after` is the generated Platform type; empty where the generated table lacks the column."""
    tables = {d: _platform(d) for d in {r['dataset'] for r in register if r['category'] != 'api_follow_up'}}
    for row in register:
        key = (row['dataset'], row['field'], row['category'])
        if row['category'] in {'serving_sql', 'api_follow_up'}:
            continue
        cols = tables[row['dataset']]
        found = type_at(cols, row['field']) if cols is not None else None
        if row['category'] in {'ddl_only', 'ppp_only', 'no_model'}:
            assert found is None, f'{key}: the generated Platform table has it'
        if row['category'] in ORDER:
            kept = set(row['before'].split(', '))
            expected = ', '.join(n for n, _ in found.fields if n in kept) if found is not None else ''
        else:
            expected = render(found, quote_fields=False) if found is not None else ''
        assert row['after'] == expected, f'{key}: after should be {expected!r}'


def test_ppp_only_columns_are_platform_removals(register):
    rows = {r['field']: r for r in register if r['dataset'] == 'evidence' and r['category'] == 'ppp_only'}
    assert sorted(rows) == sorted(PPP_ONLY)
    for field, row in rows.items():
        assert row['outcome'] == 'API removal (Platform; PPP only)', field
        assert row['after'] == '', field
        assert row['API follow-up'] == 'open', field
    platform = {n for n, _ in _platform('evidence') or []}
    every = {f.name for m in members('evidence_*', 'ppp') + members('evidence_*', 'platform')
             for f in dataset_spec(m).fields}  # fmt: skip
    assert every - platform == set(PPP_ONLY), 'generated PPP-only evidence columns differ from PPP_ONLY'


def test_superseded_rows_are_ppp_only_columns(register):
    """A non-`ppp_only` row under a PPP-only column defers its follow-up to the `ppp_only` row."""
    for row in register:
        key = (row['dataset'], row['field'], row['category'])
        top = row['field'].split('/', 1)[0].removesuffix('[]')
        under = row['dataset'] == 'evidence' and top in PPP_ONLY and row['category'] != 'ppp_only'
        assert (row['outcome'] == SUPERSEDED) == under, f'{key}: outcome {SUPERSEDED} iff under a PPP-only column'
        if under:
            assert row['after'] == '' and row['API follow-up'] == 'none', key


def test_every_sql_edit_has_a_row(register):
    """One `serving_sql` row per serving-script edit, `after` the served type."""
    with SQL_EDITS.open(newline='') as fh:
        edits = {(e['dataset'], e['field']): e for e in csv.DictReader(fh)}
    rows = {(r['dataset'], r['field']): r for r in register if r['category'] == 'serving_sql'}
    assert sorted(rows) == sorted(edits)
    for key, edit in edits.items():
        assert rows[key]['outcome'] == edit['edit'], key
        assert rows[key]['after'] == edit['served'], key


def test_every_api_follow_up_has_a_row(register):
    """One open `api_follow_up` row per line of `phase5_api_follow_ups.csv`, the text its outcome."""
    with FOLLOW_UPS.open(newline='') as fh:
        entries = {(e['dataset'], e['field']): e['follow_up'] for e in csv.DictReader(fh)}
    rows = {(r['dataset'], r['field']): r for r in register if r['category'] == 'api_follow_up'}
    assert sorted(rows) == sorted(entries)
    for key, text in entries.items():
        assert rows[key]['outcome'] == text, key
        assert rows[key]['API follow-up'] == 'open', key


def test_every_landing_change_has_a_row(register):
    """Each type change from the frozen landing DDL to the generated table, Enum values included, has a row showing
    it (or, for a nullability change, a `struct_nullability` row above it); each column order change a
    `column_order` row."""
    rows: dict[tuple[str, str], list[dict[str, str]]] = {}
    for r in register:
        rows.setdefault((r['dataset'], r['field']), []).append(r)
    shallow = build_register.shallow
    changes = build_register.landing_changes()
    missing = []
    for dataset, path, before, after in changes:
        shown = any(
            r['category'] not in ORDER and r['before'] and r['after']
            and (shallow(parse_type(r['before'])), shallow(parse_type(r['after']))) == (shallow(before), shallow(after))
            for r in rows.get((dataset, path), [])
        )  # fmt: skip
        nullability = replace(shallow(before), nullable=False) == replace(shallow(after), nullable=False)
        above = [path[:i] for i, c in enumerate(path) if c in '/[']
        structs = any(r['category'] == 'struct_nullability' for a in above for r in rows.get((dataset, a), []))
        if not shown and not (nullability and structs):
            missing.append((dataset, path, render(before), render(after)))
    assert not missing, f'landing type changes with no register row: {missing}'
    assert ('credible_sets', 'studyType') not in {(d, p) for d, p, _, _ in changes}
    for dataset, (before, after) in build_register.order_changes().items():
        order = [r for r in rows.get((dataset, ''), []) if r['category'] == 'column_order']
        assert [(r['before'], r['after']) for r in order] == [(before, after)], f'{dataset}: column_order row'


def test_cast_back_rows(register):
    """A landing-only change a serving script casts back to the landing type has follow-up `none`."""
    rows = {(r['dataset'], r['field']): r for r in register if r['category'] == 'type_unrecorded'}
    for key, entry in _cast_back().items():
        assert key in rows, f'{key}: phase5_cast_back.csv entry with no type_unrecorded row'
        assert _same(entry['served'], rows[key]['before']), key
        assert rows[key]['API follow-up'] == 'none', key


def test_every_clickhouse_marker_parses():
    """Piece 2 generates DDL from the markers, so each one must parse."""
    for name, model in discover_datasets().items():
        for path, node in walk(dataset_spec(model)):
            if node.clickhouse:
                assert parse_type(node.clickhouse), f'{name}.{path}: {node.clickhouse}'
