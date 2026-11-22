"""Build the phase 5 change register (`docs/phase5_register.csv`) from a drift CSV, run evidence and the models.

Usage, from `pos/`::

    uv run python scripts/build_register.py <drift.csv> <evidence_dir> docs/phase5_register.csv [--note <note.md>]

`drift.csv` is `scripts/schema_drift.py` output from before the piece 1 model changes (e9133802), so that rows
since resolved by a model change keep their register row. `evidence_dir` holds the `validation/` and `profile/`
parquet files of `pts.schemas.dataset.profile`. `--note` also writes the register as a Markdown note.

One row per `(dataset, field, category)`, evidence models merged. `before` is the landing DDL type, `after` the
type generated from the current model: its `ClickHouseType` or dtype; for the evidence union, nullable if any
member is nullable or lacks it; a field of a nullable struct stays Nullable. `API follow-up` is `open` when the
type changes and a serving script reads the column (column order excepted).
"""

import argparse
import csv
import importlib.util
import json
import sys
from dataclasses import replace
from functools import cache
from pathlib import Path
from typing import Any

import polars as pl
import yaml
from ot_datamodels.discovery import discover_datasets
from ot_datamodels.spec import Node, dataset_spec, walk

POS = Path(__file__).resolve().parents[1]
COLUMNS = ('dataset', 'field', 'category', 'outcome', 'before', 'after', 'evidence', 'serving read', 'API follow-up')
ORDER = frozenset({'struct_order', 'column_order'})
UNLANDED = frozenset({'model_only', 'struct_field_missing'})


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
SCRIPTS = sd._scripts(POS / 'config' / 'clickhouse' / 'scripts')


def render(t: Any) -> str:
    """The type as ClickHouse accepts it: `LowCardinality(Nullable(...))`, never the reverse."""
    if t.name == 'Array' and t.element is not None:
        text = f'Array({render(t.element)})'
    elif t.name == 'Tuple':
        text = f'Tuple({", ".join(f"{n} {render(f)}" for n, f in t.fields)})'
    elif t.args:
        text = f'{t.name}({", ".join(t.args)})'
    else:
        text = t.name
    if t.nullable:
        text = f'Nullable({text})'
    if t.low_cardinality:
        text = f'LowCardinality({text})'
    return text


def changed(category: str, before: str, after: str) -> bool:
    """Whether the column's ClickHouse type (or, for struct order, its field order) changes."""
    if category in ORDER or not before or not after:
        return before != after
    return sd.parse_type(before) != sd.parse_type(after)


def api_follow_up(category: str, before: str, after: str, read: str) -> str:
    served = read != 'no' and category != 'column_order'
    return 'open' if served and changed(category, before, after) else 'none'


@cache
def nodes(model: str) -> dict[str, Node]:
    return dict(walk(dataset_spec(MODELS[model])))


def members(dataset: str) -> list[str]:
    return [m.dataset_name for m in sd._models_for(CONFIG[dataset], MODELS) or []]


def forced(model: str, path: str) -> bool:
    """Whether a nullable struct encloses `path` without a list in between."""
    ns = nodes(model)
    parts = path.split('/')
    parts[-1] = parts[-1].removesuffix('[]')
    force = False
    for i in range(1, len(parts)):
        ancestor = '/'.join(parts[:i])
        if ancestor.endswith('[]'):
            force = False
        node = ns[ancestor]
        if node.kind == 'struct' and node.nullable:
            force = True
    return force


def union_nullable(models: list[str], path: str) -> bool:
    """Evidence union: nullable if any member is nullable there or lacks it (a list crossed absorbs absence)."""
    top = path.split('/', maxsplit=1)[0].removesuffix('[]')
    crosses_list = '[]' in path
    for m in models:
        ns = nodes(m)
        if top not in ns:
            if not crosses_list:
                return True
            continue
        if path not in ns or ns[path].nullable or forced(m, path):
            return True
    return False


def generate(node: Node, force: bool, path: str, union: list[str] | None) -> Any:
    """The ClickHouse type generated from a node: marker or dtype, Nullable inside a nullable struct."""
    if node.kind == 'list':
        assert node.element is not None
        return sd.CHType('Array', element=generate(node.element, False, f'{path}[]', union))
    if node.kind == 'struct':
        inner = force or node.nullable
        fields = tuple((f.name, generate(f.node, inner, f'{path}/{f.name}', union)) for f in node.fields)
        return sd.CHType('Tuple', fields=fields)
    base = sd.parse_type(node.clickhouse) if node.clickhouse else replace(sd.model_type(node), nullable=False)
    nullable = union_nullable(union, path) if union else node.nullable or force
    return replace(base, nullable=nullable)


def after_type(dataset: str, path: str, no_output: set[str]) -> str:
    ms = members(dataset)
    having = [m for m in ms if path in nodes(m)]
    if not having:
        return ''
    rep = next((m for m in having if m not in no_output), having[0])
    return render(generate(nodes(rep)[path], forced(rep, path), path, ms if len(ms) > 1 else None))


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
                bits.append(f'distinct {", ".join(map(str, values))}')
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
        marker = replace(sd.parse_type(node.clickhouse), nullable=False)
        if marker != replace(sd.parse_type(r['ddl_type']), nullable=False):
            return False
    return True


def outcome(category: str, dataset: str, path: str, rows: list[dict[str, str]], before: str, after: str,
            no_output: set[str]) -> str:  # fmt: skip
    evidenced = {r['model'] for r in rows if r['model']} - no_output
    if category == 'no_model':
        return 'no model (piece 3)'
    if category == 'column_order':
        return 'model order (piece 2)'
    if category == 'struct_order':
        return 'model order (piece 2)' if evidenced else 'no evidence; unchanged'
    if category in UNLANDED:
        return 'API addition (piece 2)' if evidenced else 'no evidence; unchanged'
    if category == 'ddl_only':
        return 'API removal (piece 2)' if rows[0]['serving_read'] != 'no' else 'removal (piece 2)'
    if not evidenced:
        return 'no evidence; unchanged'
    if category in {'type', 'kind', 'low_cardinality'}:
        added = _marker_added([r for r in rows if r['model'] not in no_output])
        return 'marker' if added else 'ClickHouse follows model'
    if category == 'open_vs_enum':
        return 'marker' if _marker_added(rows) else 'open string (piece 2)'
    if category == 'literal_vs_string':
        return 'String kept, no Enum'
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
        old, new = sd.parse_type(before).nullable, sd.parse_type(after).nullable
        if new and not old:
            return 'ClickHouse nullable'
        if old and not new:
            return 'ClickHouse non-null'
        return 'Nullable kept (nullable struct)'
    raise ValueError(f'unknown category {category!r}')


def build(drift: pl.DataFrame, evidence: Evidence) -> list[dict[str, str]]:
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for r in drift.iter_rows(named=True):
        groups.setdefault((r['dataset'], r['field'], r['category']), []).append(r)
    no_output = evidence.no_output
    out = []
    for (dataset, path, category), rows in groups.items():
        first = rows[0]
        read = first['serving_read']
        if category in ORDER:
            before, after = first['ddl_type'], first['model_type']
        elif category == 'no_model':
            before = after = ''
        elif category in UNLANDED:
            before, after = '', after_type(dataset, path, no_output)
            read = sd._serving(CONFIG[dataset], SCRIPTS).read(path, landed=True)
        elif category == 'ddl_only':
            before, after = first['ddl_type'], ''
        else:
            before, after = first['ddl_type'], after_type(dataset, path, no_output)
        if category == 'ddl_only':
            where = ' on every member with output' if dataset == 'evidence' else ''
            text = f'absent: strict validation clean{where}'
        elif category in {'column_order', 'no_model'}:
            text = ''
        else:
            text = evidence.text(dataset, rows, path)
        values = (
            dataset,
            path,
            category,
            outcome(category, dataset, path, rows, before, after, no_output),
            before,
            after,
            text,
            read,
            api_follow_up(category, before, after, read),
        )
        out.append(dict(zip(COLUMNS, values, strict=True)))
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
`pos/tests/test_schema_register.py` checks every row against `drift()`.

- `before`: today's landing DDL type (column or struct field order for the order rows); empty for a column \
not loaded today.
- `after`: the type piece 2 will generate from the model (marker or dtype; evidence: nullable if any \
member is nullable or lacks it; fields of a nullable struct stay Nullable); empty for a removal.
- `serving read`: `named:<script>` names the column, `pass:<script>` reads it through `SELECT *`; for a \
column not loaded today, the read it would get once loaded.
- `API follow-up`: `open` when the ClickHouse type changes from `before` to `after` and a serving script reads \
the column (column order excepted); else `none`.
"""


def _cell(value: str) -> str:
    return value.replace('|', '\\|').replace('\n', ' ')


def note(rows: list[dict[str, str]]) -> str:
    frame = pl.DataFrame(rows)
    counts = frame.group_by('outcome').agg(rows=pl.len(), open=(pl.col('API follow-up') == 'open').sum())
    n_open = sum(r['API follow-up'] == 'open' for r in rows)
    out = [NOTE_HEAD, '## Summary', '', f'{len(rows)} rows, {n_open} with API follow-up `open`.', '',
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
    parser.add_argument('drift', type=Path, help='schema_drift.py output from before the model changes')
    parser.add_argument('evidence', type=Path, help='directory with validation/ and profile/ parquet files')
    parser.add_argument('out', type=Path, help='register CSV to write')
    parser.add_argument('--note', type=Path, help='also write the register as a Markdown note')
    args = parser.parse_args()
    rows = build(pl.read_csv(args.drift, infer_schema=False), Evidence(args.evidence))
    with args.out.open('w', newline='') as fh:
        writer = csv.DictWriter(fh, COLUMNS, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    if args.note:
        args.note.write_text(note(rows))
    opened = sum(r['API follow-up'] == 'open' for r in rows)
    sys.stdout.write(f'{len(rows)} rows, {opened} open\n')


if __name__ == '__main__':
    main()
