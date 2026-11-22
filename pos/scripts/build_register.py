"""Build the phase 5 change register (`docs/phase5_register.csv`) from a drift CSV, run evidence and the models.

Usage, from `pos/`::

    uv run python scripts/build_register.py <evidence_dir> docs/phase5_register.csv \
        [--drift <drift.csv>] [--previous <register.csv>] [--note <note.md>]

Inputs:

- `evidence_dir`: the `validation/` and `profile/` parquet files of `pts.schemas.dataset.profile` for the 2612-3 run,
  copied locally from `gs://open-targets-pos-scratch/phase5/2612-3/{validation,profile}/` (7-day lifecycle).
- `--drift` (default `docs/phase5_drift_before.csv`): `scripts/schema_drift.py` output before the piece 1 model
  changes.
- `--previous` (default `docs/phase5_register_piece1.csv`): the piece 1 register; rows whose `after` differs from it
  are printed and listed in the note.
- `--note`: also write the register as a Markdown note.
- `docs/phase5_landing_before.csv` (`dataset, field, ddl_type, position`, written with `schema_drift.render`): the
  hand-written landing DDL columns at the end of piece 1. It decides whether a DDL-only column was dropped, gives the
  `before` of the `ppp_only` rows, and is compared with the generated Platform tables, Enum values included: a type
  change no row shows gets a `type_unrecorded` row, an order change of the shared columns a `column_order` row.
- `docs/phase5_sql_edits.csv` (`dataset, field, edit, table, served`): serving-script edits a generated type forces.
  One `serving_sql` row each, `before` the piece 1 landing type, `after` the type `table` serves
  (`tests/test_serving_sql.py` checks it). Where `after` is the landing type, the field's other rows are
  `superseded by serving_sql`, follow-up `none`.
- `docs/phase5_api_follow_ups.csv` (`dataset, field, follow_up`): API changes that are no ClickHouse schema diff. One
  `api_follow_up` row each, outcome the text, follow-up `open`.
- `docs/phase5_cast_back.csv` (`dataset, field, table, path, served`; `tests/test_serving_sql.py` checks the type at
  `path` in `table`): landing-only changes a serving script casts back. Where `served` is the landing type, the
  `type_unrecorded` row's follow-up is `none`.
- `docs/phase5_derived_changes.csv` (`dataset, field, table, served, note`, `note` the source column;
  `tests/test_serving_sql.py` checks the type): columns a serving script builds from an Enum column. One
  `derived_type` row each, before `Array(String)`, follow-up `open`.

The builder reads no landing DDL. One row per `(dataset, field, category)`, evidence models merged, plus a `ppp_only`
row per evidence column only PPP models have; other rows under such a column are `superseded by ppp_only`, follow-up
`none`. `before` is the landing DDL type, `after` the type `pos.clickhouse_ddl` generates for the Platform table
(evidence: `union_columns` of the Platform members), empty where that table lacks the column. `API follow-up` is
`open` when the type changes and a serving script reads the column (column order excepted).
"""

import argparse
import csv
import importlib.util
import json
import sys
from collections.abc import Iterable
from dataclasses import replace
from functools import cache
from pathlib import Path

import polars as pl
import yaml
from ot_datamodels.discovery import discover_datasets
from ot_datamodels.spec import Node, dataset_spec, walk

from pos.clickhouse_ddl import CHType, canonical, landing_columns, parse_type, render, type_at

POS = Path(__file__).resolve().parents[1]
DOCS = POS / 'docs'
PPP_ONLY = 'API removal (Platform; PPP only)'
SUPERSEDED = 'superseded by ppp_only'
SUPERSEDED_SQL = 'superseded by serving_sql'
UNRECORDED = 'type_unrecorded'
COLUMNS = ('dataset', 'field', 'category', 'outcome', 'before', 'after', 'evidence', 'serving read', 'API follow-up')
ORDER = frozenset({'struct_order', 'column_order'})
UNLANDED = frozenset({'model_only', 'struct_field_missing'})
#: Evidence that predates a pts change in this PR.
NOTES = {
    ('interaction', 'interactionId', 'type'): 'pts clears the top bit from 26.12, so ids are >= 0',
    ('interaction_evidence', 'interactionId', 'type'): 'pts clears the top bit from 26.12, so ids are >= 0',
}
#: Outcome text a row's generated type adds.
OUTCOME_NOTES = {
    ('evidence', 'datasourceId', 'literal_vs_string'): (
        '`evidence_by_variant` and `evidence_by_disease_and_target` (`ORDER BY (datasourceId, …)`) sort in the '
        "API's `Datasource` order"
    ),
}


def _load_drift():
    spec = importlib.util.spec_from_file_location('schema_drift', POS / 'scripts' / 'schema_drift.py')
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules['schema_drift'] = module
    spec.loader.exec_module(module)
    return module


sd = _load_drift()
MODELS = discover_datasets()
CONFIG = yaml.safe_load((POS / 'config' / 'datasets.yaml').read_text())['clickhouse']
CLICKHOUSE = POS / 'config' / 'clickhouse'
SCRIPTS = sd._scripts(CLICKHOUSE / 'scripts')

with (DOCS / 'phase5_landing_before.csv').open(newline='') as _fh:
    _FROZEN = sorted(csv.DictReader(_fh), key=lambda r: (r['dataset'], int(r['position'])))
LANDING = {(r['dataset'], r['field']): r['ddl_type'] for r in _FROZEN}
#: dataset -> its landing columns at 19920e46, in DDL order.
LANDING_ORDER: dict[str, list[str]] = {}
for _r in _FROZEN:
    LANDING_ORDER.setdefault(_r['dataset'], []).append(_r['field'])
with (DOCS / 'phase5_sql_edits.csv').open(newline='') as _fh:
    SQL_EDITS = list(csv.DictReader(_fh))
with (DOCS / 'phase5_api_follow_ups.csv').open(newline='') as _fh:
    API_FOLLOW_UPS = list(csv.DictReader(_fh))
with (DOCS / 'phase5_cast_back.csv').open(newline='') as _fh:
    CAST_BACK = {(r['dataset'], r['field']): r for r in csv.DictReader(_fh)}
with (DOCS / 'phase5_derived_changes.csv').open(newline='') as _fh:
    DERIVED_CHANGES = list(csv.DictReader(_fh))


def changed(category: str, before: str, after: str) -> bool:
    """Whether the column's ClickHouse type (or, for struct order, its field order) changes."""
    if category in ORDER or not before or not after:
        return before != after
    return canonical(parse_type(before)) != canonical(parse_type(after))


def api_follow_up(category: str, before: str, after: str, read: str) -> str:
    served = read != 'no' and category != 'column_order'
    return 'open' if served and changed(category, before, after) else 'none'


@cache
def nodes(model: str) -> dict[str, Node]:
    return dict(walk(dataset_spec(MODELS[model])))


@cache
def platform_table(dataset: str) -> tuple[tuple[str, CHType], ...] | None:
    """The generated Platform landing columns; None for a table with no model."""
    cols = landing_columns(CONFIG[dataset], 'platform')
    return tuple(cols) if cols is not None else None


def at(dataset: str, path: str) -> CHType | None:
    cols = platform_table(dataset)
    return type_at(cols, path) if cols is not None else None


def after_type(dataset: str, path: str) -> str:
    t = at(dataset, path)
    return render(t, quote_fields=False) if t is not None else ''


def after_order(dataset: str, path: str, before: str) -> str:
    """The generated order of the fields named in `before`."""
    t = at(dataset, path)
    kept = set(before.split(', '))
    return ', '.join(n for n, _ in t.fields if n in kept) if t is not None else ''


def _paths(cols: Iterable[tuple[str, CHType]]) -> dict[str, CHType]:
    """Every column and nested field by its register path (`a`, `a/b`, `a[]`, `a[]/b`)."""
    out: dict[str, CHType] = {}

    def visit(t: CHType, path: str) -> None:
        out[path] = t
        if t.element is not None:
            visit(t.element, f'{path}[]')
        for n, f in t.fields:
            visit(f, f'{path}/{n}')

    for name, t in cols:
        visit(t, name)
    return out


def shallow(t: CHType) -> CHType:
    """What a path's register row records: a primitive's type, Enum values explicit; an Array's or Tuple's flags."""
    if t.name in {'Array', 'Tuple'}:
        return replace(t, element=None, fields=())
    return canonical(t)


def becomes_enum(before: str, after: str) -> bool:
    """Whether a path's type turns from a non-Enum into an Enum."""
    if not before or not after:
        return False
    b, a = shallow(parse_type(before)), shallow(parse_type(after))
    return a.name.startswith('Enum') and not b.name.startswith('Enum')


def landing_changes() -> list[tuple[str, str, CHType, CHType]]:
    """(dataset, path, before, after) of each field typed differently in the frozen and the generated table."""
    out = []
    for dataset, fields in sorted(LANDING_ORDER.items()):
        cols = platform_table(dataset)
        if cols is None:
            continue
        before = _paths((f, parse_type(LANDING[dataset, f])) for f in fields)
        after = _paths(cols)
        common = sorted(before.keys() & after.keys())
        out += [(dataset, p, before[p], after[p]) for p in common if shallow(before[p]) != shallow(after[p])]
    return out


def order_changes() -> dict[str, tuple[str, str]]:
    """Dataset -> (before, after) order of the columns of both the frozen and the generated table, where it differs."""
    out = {}
    for dataset, fields in sorted(LANDING_ORDER.items()):
        generated = [n for n, _ in platform_table(dataset) or ()]
        before = [f for f in fields if f in generated]
        after = [n for n in generated if n in fields]
        if before != after:
            out[dataset] = (', '.join(before), ', '.join(after))
    return out


@cache
def ppp_only(dataset: str) -> dict[str, list[dict[str, str]]]:
    """Columns of the dataset's models absent from the Platform table -> one `{'model': ...}` per model."""
    platform = {n for n, _ in platform_table(dataset) or ()}
    found: dict[str, list[dict[str, str]]] = {}
    for model in sd._models_for(CONFIG[dataset], MODELS) or []:
        for f in dataset_spec(model).fields:
            if f.name not in platform:
                found.setdefault(f.name, []).append({'model': model.dataset_name})
    return found


class Evidence:
    """Profile and validation rows of one run, keyed by (dataset, path)."""

    def __init__(self, directory: Path) -> None:
        def read(kind: str) -> pl.DataFrame:
            parts = [pl.read_parquet(p) for p in sorted((directory / kind).glob('*.parquet'))]
            return pl.concat(parts, how='diagonal_relaxed')

        valid = read('validation')
        self.profile = {(r['dataset'], r['path']): r for r in read('profile').iter_rows(named=True)}
        self.failures = {(r['dataset'], r['column']): r for r in valid.iter_rows(named=True)}
        self.no_output = {r['dataset'] for r in valid.iter_rows(named=True) if r['check'] == 'missing'}

    def summary(self, model: str, path: str) -> str:
        if model in self.no_output:
            return 'no output in 2612-3'
        r = self.profile.get((model, path))
        if r is None:
            return 'not profiled'
        bits = []
        if r['rows'] is not None:
            bits.append(f'nulls {r["nulls"]}/{r["rows"]}')
        if r['null_elements']:
            bits.append(f'null elements {r["null_elements"]}')
        if r['min'] is not None:
            bits.append(f'min {r["min"]}, max {r["max"]}')
        if r['distinct'] is not None:
            values = json.loads(r['distinct'])
            if not values:
                bits.append('no values')
            elif len(values) <= 8:
                bits.append(f'distinct {" · ".join(map(str, values))}')
            else:
                bits.append(f'{len(values)} distinct')
        fail = self.failures.get((model, path))
        if fail:
            bits.append(f'validation {fail["check"]} {fail["failures"]}')
        return '; '.join(bits)

    def text(self, dataset: str, rows: list[dict[str, str]], path: str) -> str:
        models = sorted({r['model'] for r in rows if r['model']})
        if dataset == 'evidence':
            return ' | '.join(f'{m.removeprefix("evidence_")}: {self.summary(m, path)}' for m in models)
        return self.summary(models[0], path) if models else ''


def _marker_added(rows: list[dict[str, str]]) -> bool:
    """Whether every drift row's model now carries a marker equal to the DDL type."""
    for r in rows:
        if not r['model']:
            continue
        node = nodes(r['model']).get(r['field'])
        if node is None or not node.clickhouse:
            return False
        marker = replace(parse_type(node.clickhouse), nullable=False)
        if marker != replace(parse_type(r['ddl_type']), nullable=False):
            return False
    return True


def outcome(category: str, dataset: str, path: str, rows: list[dict[str, str]], before: str, after: str,
            no_output: set[str]) -> str:  # fmt: skip
    evidenced = {r['model'] for r in rows if r['model']} - no_output
    if category == 'no_model':
        return 'no model (piece 3)'
    if category == 'ppp_only':
        return PPP_ONLY
    if category == 'column_order':
        return 'model order (piece 2)'
    if category == 'struct_order':
        return 'model order (piece 2)' if evidenced else 'no evidence; unchanged'
    if category in UNLANDED:
        return 'API addition (piece 2)' if evidenced else 'no evidence; unchanged'
    if category == 'ddl_only' and (dataset, path) not in LANDING:
        return 'API removal (dropped)'
    if category == 'ddl_only':
        return 'API removal (piece 2)' if rows[0]['serving_read'] != 'no' else 'removal (piece 2)'
    if not evidenced:
        return 'no evidence; unchanged'
    if category in {'type', 'kind', 'low_cardinality'}:
        added = _marker_added([r for r in rows if r['model'] not in no_output])
        return 'marker' if added else 'ClickHouse follows model'
    if category == 'open_vs_enum':
        if _marker_added(rows):
            return 'marker'
        return 'Enum in the model' if after and parse_type(after).name.startswith('Enum') else 'open string (piece 2)'
    if category == 'literal_vs_string':
        return 'String → Enum' if after and parse_type(after).name.startswith('Enum') else 'String kept, no Enum'
    if category == 'list_nullability':
        return 'null list loads as [] (model unchanged)'
    if category == 'struct_nullability':
        return 'null struct: Tuple fields Nullable (model unchanged)'
    if category == 'nullability':
        was = {r['model']: r['model_nullable'] == 'true' for r in rows}
        if any(nodes(m)[path].nullable and not was[m] for m in evidenced if path in nodes(m)):
            return 'model nullable'
        if dataset == 'evidence':
            return 'evidence union (piece 2)'
        old, new = parse_type(before).nullable, parse_type(after).nullable
        if new and not old:
            return 'ClickHouse nullable'
        if old and not new:
            return 'ClickHouse non-null'
        return 'Nullable kept (nullable struct)'
    raise ValueError(f'unknown category {category!r}')


def sql_edit(edit: dict[str, str], evidence: Evidence) -> dict[str, str]:
    """The `serving_sql` row of one serving-script edit."""
    dataset, path, after = edit['dataset'], edit['field'], edit['served']
    before = LANDING[dataset, path]
    read = sd._serving(CONFIG[dataset], SCRIPTS).read(path, landed=True)
    models = sd._models_for(CONFIG[dataset], MODELS) or []
    text = evidence.text(dataset, [{'model': m.dataset_name} for m in models], path)
    values = (dataset, path, 'serving_sql', edit['edit'], before, after, text, read,
              api_follow_up('serving_sql', before, after, read))  # fmt: skip
    return dict(zip(COLUMNS, values, strict=True))


def follow_up_row(entry: dict[str, str]) -> dict[str, str]:
    """The `api_follow_up` row of one API change that is no schema diff."""
    values = (entry['dataset'], entry['field'], 'api_follow_up', entry['follow_up'], '', '', '', '', 'open')
    return dict(zip(COLUMNS, values, strict=True))


def derived_row(entry: dict[str, str]) -> dict[str, str]:
    """The `derived_type` row of one serving-script column built from an Enum column."""
    dataset, field, source = entry['dataset'], entry['field'], entry['note']
    read = sd._serving(CONFIG[dataset], SCRIPTS).read(field, landed=True)
    values = (dataset, field, 'derived_type', f'derived from Enum column {source}', 'Array(String)', entry['served'],
              '', read, 'open')  # fmt: skip
    return dict(zip(COLUMNS, values, strict=True))


def same(a: str, b: str) -> bool:
    return canonical(parse_type(a)) == canonical(parse_type(b))


@cache
def kept_by_sql_edit() -> frozenset[tuple[str, str]]:
    """(dataset, field) of each serving-script edit that serves the landing type."""
    kept = (e for e in SQL_EDITS if same(e['served'], LANDING[e['dataset'], e['field']]))
    return frozenset((e['dataset'], e['field']) for e in kept)


def cast_back(dataset: str, path: str, before: CHType) -> dict[str, str] | None:
    """The `phase5_cast_back.csv` entry of a field whose serving table serves the landing type."""
    entry = CAST_BACK.get((dataset, path))
    return entry if entry is not None and canonical(parse_type(entry['served'])) == canonical(before) else None


def _ancestors(path: str) -> list[str]:
    """`a`, `a[]` and `a[]/b` for `a[]/b/c`."""
    return [path[:i] for i in range(1, len(path)) if path[i] == '/' or path.startswith('[]', i)]


def _recorded(rows: dict[tuple[str, str], list[dict[str, str]]], dataset: str, path: str, before: CHType,
              after: CHType) -> bool:  # fmt: skip
    """Whether a row at the path shows the change, or a `struct_nullability` row above it a nullability change."""
    for r in rows.get((dataset, path), []):
        if r['category'] in ORDER or not r['before'] or not r['after']:
            continue
        if shallow(parse_type(r['before'])) == shallow(before) and shallow(parse_type(r['after'])) == shallow(after):
            return True
    if replace(shallow(before), nullable=False) != replace(shallow(after), nullable=False):
        return False
    return any(r['category'] == 'struct_nullability' for a in _ancestors(path) for r in rows.get((dataset, a), []))


def describe(before: CHType, after: CHType) -> str:
    """The change from `before` to `after`, in words."""
    b, a = shallow(before), shallow(after)
    word_b, word_a = ('Enum' if t.name.startswith('Enum') else t.name for t in (b, a))
    parts = []
    if word_b == word_a == 'Enum':
        labels = [sorted(e.rsplit(' = ', 1)[0] for e in t.args) for t in (b, a)]
        parts.append('Enum renumbered' if labels[0] == labels[1] else 'Enum labels changed')
    elif (b.name, b.args) != (a.name, a.args):
        parts.append(f'{word_b} → {word_a}')
    if b.nullable != a.nullable:
        parts.append('ClickHouse nullable' if a.nullable else 'ClickHouse non-null')
    if b.low_cardinality != a.low_cardinality:
        parts.append('LowCardinality added' if a.low_cardinality else 'LowCardinality dropped')
    return '; '.join(parts)


def unrecorded(rows: list[dict[str, str]], evidence: Evidence) -> list[dict[str, str]]:
    """A `type_unrecorded` row per landing type change no row shows, a `column_order` row per unrecorded order."""
    keyed: dict[tuple[str, str], list[dict[str, str]]] = {}
    for r in rows:
        keyed.setdefault((r['dataset'], r['field']), []).append(r)
    out = []
    for dataset, path, b, a in landing_changes():
        if _recorded(keyed, dataset, path, b, a):
            continue
        before, after = sd.render(b), after_type(dataset, path)
        read = sd._serving(CONFIG[dataset], SCRIPTS).read(path, landed=True)
        models = sd._models_for(CONFIG[dataset], MODELS) or []
        text = evidence.text(dataset, [{'model': m.dataset_name} for m in models], path)
        result, follow_up = describe(b, a), api_follow_up(UNRECORDED, before, after, read)
        if (entry := cast_back(dataset, path, b)) is not None:
            result, follow_up = f'{result}; landing only: {entry["table"]} casts back', 'none'
        values = (dataset, path, UNRECORDED, result, before, after, text, read, follow_up)
        out.append(dict(zip(COLUMNS, values, strict=True)))
    for dataset, (before, after) in order_changes().items():
        if any(r['category'] == 'column_order' for r in keyed.get((dataset, ''), [])):
            continue
        read = sd._serving(CONFIG[dataset], SCRIPTS).read('', landed=True)
        values = (dataset, '', 'column_order', outcome('column_order', dataset, '', [], before, after, set()),
                  before, after, '', read, api_follow_up('column_order', before, after, read))  # fmt: skip
        out.append(dict(zip(COLUMNS, values, strict=True)))
    return out


def build(drift: pl.DataFrame, evidence: Evidence) -> list[dict[str, str]]:
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for r in drift.iter_rows(named=True):
        groups.setdefault((r['dataset'], r['field'], r['category']), []).append(r)
    for dataset, entry in CONFIG.items():
        if 'models' not in entry:
            continue
        serving = sd._serving(entry, SCRIPTS)
        for col, ms in ppp_only(dataset).items():
            d = LANDING.get((dataset, col), '')
            cells = {'ddl_type': d, 'serving_read': serving.read(col, landed=bool(d))}
            groups[dataset, col, 'ppp_only'] = [m | cells for m in ms]
    no_output = evidence.no_output
    out = [sql_edit(e, evidence) for e in SQL_EDITS]
    out += [follow_up_row(f) for f in API_FOLLOW_UPS]
    out += [derived_row(d) for d in DERIVED_CHANGES]
    for (dataset, path, category), rows in groups.items():
        first = rows[0]
        read = first['serving_read']
        if category in ORDER:
            before = first['ddl_type']
            after = after_order(dataset, path, before)
        elif category == 'no_model':
            before = after = ''
        elif category in UNLANDED:
            before, after = '', after_type(dataset, path)
            read = sd._serving(CONFIG[dataset], SCRIPTS).read(path, landed=True)
        elif category == 'ddl_only':
            before, after = first['ddl_type'], ''
        else:
            before, after = first['ddl_type'], after_type(dataset, path)
        if category == 'ddl_only':
            where = ' on every member with output' if dataset == 'evidence' else ''
            text = f'absent: strict validation clean{where}'
        elif category in {'column_order', 'no_model'}:
            text = ''
        else:
            text = evidence.text(dataset, rows, path)
        if (dataset, path, category) in NOTES:
            text = f'{text}; {NOTES[dataset, path, category]}'
        top = path.split('/', 1)[0].removesuffix('[]')
        if category != 'ppp_only' and top in ppp_only(dataset):
            result, follow_up = SUPERSEDED, 'none'
        elif category != 'serving_sql' and (dataset, path) in kept_by_sql_edit():
            result, follow_up = SUPERSEDED_SQL, 'none'
        else:
            result = outcome(category, dataset, path, rows, before, after, no_output)
            if category not in ORDER and becomes_enum(before, after) and 'Enum' not in result:
                result = f'{result}; String → Enum'
            follow_up = api_follow_up(category, before, after, read)
        if (dataset, path, category) in OUTCOME_NOTES:
            result = f'{result}; {OUTCOME_NOTES[dataset, path, category]}'
        values = (dataset, path, category, result, before, after, text, read, follow_up)
        out.append(dict(zip(COLUMNS, values, strict=True)))
    out += unrecorded(out, evidence)
    return sorted(out, key=lambda r: (r['dataset'], r['field'], r['category']))


NOTE_HEAD = """\
---
created: 2026-11-22
status: draft
area: Work
tags: [pipeline, pos, datamodels, clickhouse, spec]
---

# POS phase 5: API change register

Every schema change of phase 5 piece 1 ([[POS-Phase5-Drift-Design]]), one row per disagreement between the \
dataset models and the ClickHouse landing DDL, after evidence models are merged (one row per \
`(dataset, field, category)`). Data evidence: [[POS-Phase5-Evidence]] (run `do/platform-2612-3`). \
The same rows are in `pos/docs/phase5_register.csv`, written by `pos/scripts/build_register.py`; \
`pos/tests/test_schema_register.py` checks every `after` against the generator.

- `before`: today's landing DDL type (column or struct field order for the order rows); empty for a column \
not loaded today.
- `after`: the type `pos.clickhouse_ddl` generates for the Platform table (marker or dtype; fields of a nullable \
struct stay Nullable; evidence: `union_columns` of the Platform evidence models, Nullable where any is nullable or \
lacks the column); empty where the generated Platform table lacks the column (a removal, or a `ppp_only` evidence \
column).
- `serving read`: `named:<script>` names the column, `pass:<script>` reads it through `SELECT *`; for a \
column not loaded today, the read it would get once loaded.
- `serving_sql`: a serving-script edit a generated type forced; `after` is the type the serving table serves.
- `api_follow_up`: an API change that is no schema diff; the outcome says what.
- `API follow-up`: `open` when the ClickHouse type changes from `before` to `after` and a serving script reads \
the column (column order excepted); else `none`.
"""


HINTS = """
## Decisions (David, 2026-11-22)

- `interactionId`: pts clears the hash's top bit, so ids are non-negative Int64; the models say `ge=0` and keep a \
`UInt64` marker. Every id changes from 2612-3; the polars port had already changed them all since the last release.
- `directLocationIds` / `indirectLocationIds`: dropped from the disease landing DDL and `disease.sql` (pts writes no \
locations; the API served empty arrays). The API drops them; disease locations come back as their own dataset (#240).

## Piece 2 hints

- `interaction.count` is UInt8 with observed max 227 and no `le` guard: ClickHouse would wrap at 256. Consider \
`le=255` in the model.
- `LowCardinality` markers hold the non-null type. On a nullable field emit `LowCardinality(Nullable(String))`; \
`Nullable(LowCardinality(..))` is rejected by ClickHouse. Affects pharmacogenomics `drugs[].drugId`, study \
`publicationJournal` and `condition`, interaction `speciesB.scientificName`.
- Fields of a nullable struct stay Nullable (clinical_report `trialSponsor/*`, clinical_target `diseases[]`): a \
Tuple cannot be NULL, so a non-null field would turn a null struct into `''`.
- The 3 `chromosome` Enums stay open strings (API open); `transcript.strand` and `exons[].strand` get an Enum8 marker. \
Superseded by the Enums below.
- Rows with no 2612-3 evidence take `after` from the model; check those datasets' next outputs first (otar, the \
assays and validationReadouts of ot_crispr_validation and encore, the encore struct order).
- Model-nullable scalars all have nulls in the data, so they stay nullable and ClickHouse becomes Nullable.
- `struct_order` rows are `open` where served (Tuple field order is part of the type); `column_order` is `none`.
"""


ENUMS = """
## Enums (David, 2026-11-22)

Closed vocabularies are Enums in the models (`ot_datamodels.vocabularies`), so ClickHouse gets an `Enum8` in the \
vocabulary's order: today's ClickHouse Enums (chromosome, study type, interaction source, fine-mapping method), Open \
Targets' own vocabularies (datasource, datatype, clinical stage, provider, origin...) and fixed external standards \
(ClinicalTrials.gov, GO, HPO, ChEMBL, Ensembl, gnomAD, ClinPGx). A column that becomes an Enum has a row with \
follow-up `open` where served, and its outcome says `String → Enum`.

- API enums that disagree with the data's vocabulary (`Provider` lacks `FDA` and spells `CHEMBL`; `Chromosome` is \
`chr`-prefixed) have an `api_follow_up` row each, from \
`pos/docs/phase5_api_follow_ups.csv`.
- No API change for `aggregationType`: `datatypeId` is only in `association_by_datatype_*`, which ClickHouse does \
not serve; the API's `AggregationType` reads `association_time_series`, built from `association_by_datasource_*` \
(`overall`, `datasourceId`).
- `evidence.sql` `ORDER BY` `datasourceId` (and any sort key on an Enum column) now sorts by the Enum's order (the \
API's `Datasource` order), not alphabetically.
- Served via the hand-written `targets` table; type unchanged until piece 3: `homology.homologyType`, \
`target_tractability.modality`, \
`target.genomicLocation.chromosome`, `target.go[].evidence`, `target.go[].aspect`, \
`target.constraint[].constraintType`. pts `target_view` keeps writing what `targets.sql` expects.
"""


def _cell(value: str) -> str:
    return value.replace('|', '\\|').replace('\n', ' ')


PIECE2 = """
## Piece 2

`after` is now the generated Platform type. The Platform evidence table drops the 15 columns only PPP models have \
(`ppp_only` rows, outcome `API removal (Platform; PPP only)`, which carry the API follow-up); the other rows under \
those columns are `superseded by ppp_only`, follow-up `none`. Rows of a field whose `serving_sql` edit serves the \
landing type are `superseded by serving_sql`, follow-up `none`. `type_unrecorded` rows are type changes from the \
landing DDL that no piece 1 row showed (an Enum against a closed `Literal`, Enum values); where a serving script \
casts the column back to the landing type, the outcome says so and the follow-up is `none`. Rows whose `after` \
changed from the piece 1 register:
"""

Changes = list[tuple[dict[str, str], dict[str, str]]]


def after_changes(rows: list[dict[str, str]], previous: Path) -> Changes:
    """(previous row, row) for each row whose `after` changed."""
    with previous.open(newline='') as fh:
        old = {(r['dataset'], r['field'], r['category']): r for r in csv.DictReader(fh)}
    keyed = ((old.get((r['dataset'], r['field'], r['category'])), r) for r in rows)
    return [(o, r) for o, r in keyed if o is not None and o['after'] != r['after']]


def note(rows: list[dict[str, str]], changes: Changes) -> str:
    frame = pl.DataFrame(rows)
    counts = frame.group_by('outcome').agg(rows=pl.len(), open=(pl.col('API follow-up') == 'open').sum())
    n_open = sum(r['API follow-up'] == 'open' for r in rows)
    out = [NOTE_HEAD, HINTS, ENUMS, PIECE2, '| dataset | field | category | outcome | after | API follow-up |',
           '| --- | --- | --- | --- | --- | --- |']  # fmt: skip
    for o, r in changes:
        after = f'`{o["after"]}` → `{r["after"]}`' if r['after'] else f'`{o["after"]}` → empty'
        values = [r['dataset'], f'`{r["field"]}`' if r['field'] else '', r['category'], r['outcome'], after,
                  f'{o["API follow-up"]} → {r["API follow-up"]}']  # fmt: skip
        out.append(f'| {" | ".join(_cell(v) for v in values)} |')
    out += ['', '## Summary', '', f'{len(rows)} rows, {n_open} with API follow-up `open`.', '',
            '| outcome | rows | open |', '| --- | --- | --- |']  # fmt: skip
    out += [f'| {r["outcome"]} | {r["rows"]} | {r["open"]} |' for r in counts.sort('outcome').iter_rows(named=True)]
    out += ['', '## Register', '', f'| {" | ".join(COLUMNS)} |', f'|{" --- |" * len(COLUMNS)}']
    for r in rows:
        code = {k: f'`{r[k]}`' if r[k] else '' for k in ('field', 'before', 'after')}
        values = [r['dataset'], code['field'], r['category'], r['outcome'], code['before'], code['after'],
                  r['evidence'], r['serving read'], r['API follow-up']]  # fmt: skip
        out.append(f'| {" | ".join(_cell(v) for v in values)} |')
    return '\n'.join(out) + '\n'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n', 1)[0])
    parser.add_argument('evidence', type=Path, help='directory with validation/ and profile/ parquet files')
    parser.add_argument('out', type=Path, help='register CSV to write')
    parser.add_argument(
        '--drift', type=Path, default=DOCS / 'phase5_drift_before.csv', help='pre-change drift snapshot'
    )
    parser.add_argument(
        '--previous', type=Path, default=DOCS / 'phase5_register_piece1.csv', help='register to list changes against'
    )
    parser.add_argument('--note', type=Path, help='also write the register as a Markdown note')
    args = parser.parse_args()
    rows = build(pl.read_csv(args.drift, infer_schema=False), Evidence(args.evidence))
    changes = after_changes(rows, args.previous)
    with args.out.open('w', newline='') as fh:
        writer = csv.DictWriter(fh, COLUMNS, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    if args.note:
        args.note.write_text(note(rows, changes))
    for o, r in changes:
        sys.stdout.write(f'{r["dataset"]}\t{r["field"]}\t{r["category"]}\t{o["after"]} -> {r["after"]}\t'
                         f'{o["API follow-up"]} -> {r["API follow-up"]}\n')  # fmt: skip
    opened = sum(r['API follow-up'] == 'open' for r in rows)
    sys.stdout.write(f'{len(rows)} rows, {opened} open\n')


if __name__ == '__main__':
    main()
