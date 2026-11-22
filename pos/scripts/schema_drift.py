"""List every disagreement between the dataset models and the ClickHouse landing DDL.

Usage: `uv run python scripts/schema_drift.py <out.csv>` from `pos/`. Reads code and config only.

Compares the entries with a hand-written `preload_script`: a landing table maps to its entry's `model`, or to
every model matching its `models` glob; an entry with neither is listed as `no_model`. Entries generated from
their model (`model`/`models` and no `preload_script`) have no hand-written DDL and are skipped.
"""

import re
import sys
from collections.abc import Iterator
from dataclasses import dataclass, replace
from fnmatch import fnmatch
from pathlib import Path

import polars as pl
import yaml
from ot_datamodels.base import DatasetModel
from ot_datamodels.discovery import discover_datasets
from ot_datamodels.spec import Node, dataset_spec

from pos import clickhouse_ddl as ddl
from pos.clickhouse_ddl import ENUMS, CHType, model_type, parse_table, parse_type

CATEGORIES = (
    'model_only',
    'ddl_only',
    'struct_field_missing',
    'type',
    'kind',
    'low_cardinality',
    'nullability',
    'list_nullability',
    'struct_nullability',
    'literal_vs_string',
    'open_vs_enum',
    'struct_order',
    'column_order',
)
COLUMNS = (
    'category',
    'dataset',
    'model',
    'field',
    'model_type',
    'model_clickhouse',
    'ddl_type',
    'model_nullable',
    'ddl_nullable',
    'serving_read',
)

_INTS = frozenset({f'{s}Int{w}' for s in ('', 'U') for w in (8, 16, 32, 64, 128, 256)})
_FLOATS = frozenset({'Float32', 'Float64'})


def render(t: CHType) -> str:
    """The type as the drift CSVs spell it: `Nullable` outermost, Tuple field names bare."""
    return ddl.render(t, nullable_outermost=True, quote_fields=False)


# models


def _enum_labels(t: CHType) -> set[str]:
    return {a.split(' = ')[0].strip("'") for a in t.args}


def _enum_values(t: CHType) -> set[int]:
    return {int(a.split(' = ')[1]) for a in t.args if ' = ' in a}


def _type_categories(node: Node, d: CHType) -> list[str]:
    """type/kind/low_cardinality/literal_vs_string/open_vs_enum of a primitive against its DDL type."""
    marker = node.clickhouse
    m = replace(parse_type(marker) if marker else model_type(node), nullable=False)
    d = replace(d, nullable=False)
    allowed = node.constraints.allowed or ()
    allowed_str = bool(allowed) and all(isinstance(v, str) for v in allowed)
    cats = []
    if not (marker and m == d):
        if m.low_cardinality != d.low_cardinality:
            cats.append('low_cardinality')
        a, b = replace(m, low_cardinality=False), replace(d, low_cardinality=False)
        if b.name in ENUMS:
            if allowed_str:
                if set(allowed) != _enum_labels(b):
                    cats.append('type')
            elif allowed:
                if set(allowed) != _enum_values(b):
                    cats.append('type')
            elif a.name not in ENUMS:
                cats.append('open_vs_enum')
                if a.name != 'String':
                    cats.append('kind')
            elif a != b:
                cats.append('type')
        elif allowed_str and b.name == 'String':
            pass
        elif a != b:
            same = {a.name, b.name} <= _INTS or {a.name, b.name} <= _FLOATS
            cats.append('type' if same else 'kind')
    if allowed_str and d.name == 'String':
        cats.append('literal_vs_string')
    return cats


# serving scripts


_NOT_ALIASES = frozenset({
    'where', 'on', 'using', 'left', 'right', 'inner', 'outer', 'full', 'cross', 'join', 'array', 'group', 'order',
    'limit', 'union', 'settings', 'format', 'final', 'prewhere', 'having', 'as',
})  # fmt: skip


def _reads_star(text: str, table: str) -> bool:
    """Whether a script selects `*`, `* EXCEPT ...`, `<table>.*` or `<alias>.*` from `table`."""
    t = re.escape(table)
    # a bare * item of a select list, then the FROM of that same select
    for star in re.finditer(r'(?:\bselect|,)\s*\*', text, re.IGNORECASE):
        depth = 0
        for m in re.finditer(r'\(|\)|;|\bfrom\b', text[star.end() :], re.IGNORECASE):
            word = m.group().lower()
            if word == '(':
                depth += 1
            elif word == ')':
                depth -= 1
            if depth < 0 or word == ';':
                break
            if word == 'from' and depth == 0:
                if re.match(rf'\s+{t}\b', text[star.end() + m.end() :]):
                    return True
                break
    aliases = {table}
    for m in re.finditer(rf'\b(?:from|join)\s+{t}(?:\s+as)?\s+(\w+)', text, re.IGNORECASE):
        if m.group(1).lower() not in _NOT_ALIASES:
            aliases.add(m.group(1))
    return any(re.search(rf'(?<![\w.]){re.escape(a)}\.\*', text) for a in aliases)


@dataclass(frozen=True)
class _Serving:
    """Every script in the scripts dir that mentions the landing table, name -> comment-free text."""

    table: str
    scripts: tuple[tuple[str, str], ...]

    def read(self, path: str, *, landed: bool) -> str:
        """`named:<s>` for each script naming the field or its column, `pass:<s>` for each other one reading `*`."""
        if not landed:
            return 'no'
        parts = [p.removesuffix('[]') for p in path.split('/')] if path else []
        words = {parts[0], parts[-1]} if parts else set()
        named = [n for n, text in self.scripts if any(re.search(rf'\b{re.escape(w)}\b', text) for w in words)]
        passed = [n for n, text in self.scripts if n not in named and _reads_star(text, self.table)]
        hits = [f'named:{n}' for n in named] + [f'pass:{n}' for n in passed]
        return ';'.join(hits) or 'no'


# diff


@dataclass
class _Diff:
    dataset: str
    model: str
    serving: _Serving
    rows: list[dict[str, str]]

    def add(self, category: str, path: str, cells: dict[str, str], *, landed: bool = True) -> None:
        """`cells` fills model_type, model_clickhouse, ddl_type, model_nullable and ddl_nullable."""
        row = dict.fromkeys(COLUMNS, '') | cells
        row |= {'category': category, 'dataset': self.dataset, 'model': self.model, 'field': path}
        self.rows.append(row | {'serving_read': self.serving.read(path, landed=landed)})

    def missing(self, category: str, path: str, node: Node) -> None:
        self.add(category, path, _model_cells(node), landed=False)

    def node(self, path: str, node: Node, d: CHType) -> None:
        cells = _model_cells(node) | _ddl_cells(d)
        container = {'list': 'Array', 'struct': 'Tuple'}.get(node.kind)
        mismatch = d.name != container if container else d.name in {'Array', 'Tuple'}
        if mismatch:
            self.add('kind', path, cells)
            return
        if node.kind == 'list':
            assert node.element is not None and d.element is not None
            if node.nullable:
                self.add('list_nullability', path, cells)
            self.node(f'{path}[]', node.element, d.element)
            return
        if node.kind == 'struct':
            if node.nullable:
                self.add('struct_nullability', path, cells)
            self.fields(path, [(f.name, f.node) for f in node.fields], dict(d.fields), 'struct')
            return
        for category in _type_categories(node, d):
            self.add(category, path, cells)
        if node.nullable != d.nullable:
            self.add('nullability', path, cells)

    def fields(self, path: str, model: list[tuple[str, Node]], ddl: dict[str, CHType], level: str) -> None:
        prefix = f'{path}/' if path else ''
        names = {name for name, _ in model}
        for name, node in model:
            if name in ddl:
                self.node(f'{prefix}{name}', node, ddl[name])
            else:
                self.missing('model_only' if level == 'column' else 'struct_field_missing', f'{prefix}{name}', node)
        for name, d in ddl.items():
            if name not in names:
                self.add('ddl_only', f'{prefix}{name}', _ddl_cells(d))
        order_m = [n for n, _ in model if n in ddl]
        order_d = [n for n in ddl if n in names]
        if order_m != order_d:
            self.add(f'{level}_order', path, {'model_type': ', '.join(order_m), 'ddl_type': ', '.join(order_d)})


def _model_cells(node: Node) -> dict[str, str]:
    return {
        'model_type': render(model_type(node)),
        'model_clickhouse': node.clickhouse or '',
        'model_nullable': _b(node.nullable),
    }


def _ddl_cells(d: CHType) -> dict[str, str]:
    return {'ddl_type': render(d), 'ddl_nullable': _b(d.nullable)}


def _b(value: bool) -> str:
    return 'true' if value else 'false'


def _scripts(scripts_dir: Path) -> tuple[tuple[str, str], ...]:
    return tuple((p.name, re.sub(r'--[^\n]*', '', p.read_text())) for p in sorted(scripts_dir.glob('*.sql')))


def _serving(entry: dict, scripts: tuple[tuple[str, str], ...]) -> _Serving:
    table = entry.get('table', '')
    pattern = re.compile(rf'\b{re.escape(table)}\b')
    return _Serving(table, tuple((n, text) for n, text in scripts if table and pattern.search(text)))


def _models_for(entry: dict, models: dict[str, type[DatasetModel]]) -> list[type[DatasetModel]] | None:
    if 'models' in entry:
        return [m for name, m in sorted(models.items()) if fnmatch(name, entry['models'])] or None
    model = models.get(entry.get('model', ''))
    return [model] if model else None


def _dataset_rows(dataset: str, entry: dict, schema_dir: Path, scripts: tuple[tuple[str, str], ...],
                  models: dict[str, type[DatasetModel]]) -> Iterator[dict[str, str]]:  # fmt: skip
    serving = _serving(entry, scripts)
    matched = _models_for(entry, models)
    if matched is None:
        yield dict.fromkeys(COLUMNS, '') | {'category': 'no_model', 'dataset': dataset, 'serving_read': 'no'}
        return
    ddl = parse_table((schema_dir / Path(entry['preload_script']).name).read_text())
    union = len(matched) > 1 or 'models' in entry
    rows: list[dict[str, str]] = []
    for model in matched:
        spec = dataset_spec(model)
        diff = _Diff(dataset, model.dataset_name, serving, rows)
        columns = [(f.name, f.node) for f in spec.fields]
        if union:
            # one DDL covers many models: no per-model ddl_only or column order
            for name, node in columns:
                if name in ddl:
                    diff.node(name, node, ddl[name])
                else:
                    diff.missing('model_only', name, node)
        else:
            diff.fields('', columns, ddl, 'column')
    if union:
        names = {f.name for m in matched for f in dataset_spec(m).fields}
        diff = _Diff(dataset, '', serving, rows)
        for name, d in ddl.items():
            if name not in names:
                diff.add('ddl_only', name, _ddl_cells(d))
    yield from rows


def drift(datasets_yaml: Path, schema_dir: Path, scripts_dir: Path, *,
          models: dict[str, type[DatasetModel]] | None = None) -> pl.DataFrame:  # fmt: skip
    """One row per disagreement between each hand-written landing table and its model(s); `no_model` for the rest.

    Only entries with a `preload_script` have hand-written DDL; derived entries are skipped.
    """
    models = discover_datasets() if models is None else models
    config = yaml.safe_load(datasets_yaml.read_text())['clickhouse']
    scripts = _scripts(scripts_dir)
    rows = [
        row
        for dataset, entry in config.items()
        if 'preload_script' in entry and not entry.get('derived')
        for row in _dataset_rows(dataset, entry, schema_dir, scripts, models)
    ]
    return pl.DataFrame(rows, schema=dict.fromkeys(COLUMNS, pl.String))


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit('usage: schema_drift.py <out.csv>')
    pos = Path(__file__).resolve().parents[1]
    clickhouse = pos / 'config' / 'clickhouse'
    frame = drift(pos / 'config' / 'datasets.yaml', clickhouse / 'schema', clickhouse / 'scripts')
    frame.write_csv(sys.argv[1])
    read = pl.col('serving_read').str.extract(r'^(\w+)')  # first hit: named before pass
    counts = frame.group_by('category').agg(
        items=pl.len(), **{r: (read == r).sum() for r in ('named', 'pass', 'no')}
    )
    with pl.Config(tbl_rows=-1, tbl_hide_column_data_types=True, tbl_hide_dataframe_shape=True):
        sys.stdout.write(f'{counts.sort("category")}\n')


if __name__ == '__main__':
    main()
