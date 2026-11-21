"""List every disagreement between the dataset models and the ClickHouse landing DDL.

Usage: `uv run python scripts/schema_drift.py <out.csv>` from `pos/`. Reads code and config only.

A landing table maps to the model whose `dataset_name` is the last segment of its `output/...`
`input_dir`; `evidence` (a glob over `output/evidence_*`) maps to every `evidence_*` model. Tables fed
from `view/...` have no model and are listed as `no_model`.
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

_ENUMS = frozenset({'Enum', 'Enum8', 'Enum16'})
_INTS = frozenset({f'{s}Int{w}' for s in ('', 'U') for w in (8, 16, 32, 64, 128, 256)})
_FLOATS = frozenset({'Float32', 'Float64'})
_POLARS_NAMES = {'Boolean': 'Bool'}
_TOKEN = re.compile(r"\s*(?:`([^`]*)`|('(?:[^'\\]|\\.)*')|(-?\d+(?:\.\d+)?)|([A-Za-z_]\w*)|(\S))")
_CREATE = re.compile(r'create\s+table\s+(?:if\s+not\s+exists\s+)?[\w.`]+\s*\(', re.IGNORECASE)


@dataclass(frozen=True)
class CHType:
    """A ClickHouse type: a primitive (`args` hold Enum entries or a length), an Array or a Tuple."""

    name: str
    nullable: bool = False
    low_cardinality: bool = False
    args: tuple[str, ...] = ()
    element: 'CHType | None' = None
    fields: tuple[tuple[str, 'CHType'], ...] = ()


# DDL parsing


def _tokens(text: str) -> list[tuple[str, str]]:
    """(kind, value) pairs; kind is `name` (identifier or backticked), `str`, `num` or `sym`."""
    out = []
    for m in _TOKEN.finditer(text):
        quoted, string, number, ident, sym = m.groups()
        if quoted is not None:
            out.append(('name', quoted))
        elif string is not None:
            out.append(('str', string))
        elif number is not None:
            out.append(('num', number))
        elif ident is not None:
            out.append(('name', ident))
        elif sym is not None:
            out.append(('sym', sym))
    return out


class _Parser:
    def __init__(self, text: str) -> None:
        self.tokens = _tokens(text)
        self.i = 0

    def peek(self, ahead: int = 0) -> tuple[str, str]:
        j = self.i + ahead
        return self.tokens[j] if j < len(self.tokens) else ('end', '')

    def take(self, kind: str | None = None, value: str | None = None) -> str:
        k, v = self.peek()
        if (kind and k != kind) or (value and v != value):
            raise ValueError(f'expected {value or kind} at token {self.i}, got {v!r}')
        self.i += 1
        return v

    def at(self, value: str) -> bool:
        return self.peek()[1] == value

    def items(self, item):
        """`(item, item, ...)`, with an optional trailing comma."""
        self.take('sym', '(')
        out = []
        while not self.at(')'):
            out.append(item())
            if not self.at(')'):
                self.take('sym', ',')
        self.take('sym', ')')
        return out

    def type(self) -> CHType:
        name = self.take('name')
        if name in {'Nullable', 'LowCardinality'}:
            self.take('sym', '(')
            inner = self.type()
            self.take('sym', ')')
            flag = 'nullable' if name == 'Nullable' else 'low_cardinality'
            return replace(inner, **{flag: True})
        if name == 'Array':
            self.take('sym', '(')
            element = self.type()
            self.take('sym', ')')
            return CHType('Array', element=element)
        if name == 'Tuple':
            return CHType('Tuple', fields=tuple(self.items(self.tuple_field)))
        if name in _ENUMS:
            return CHType(name, args=tuple(self.items(self.enum_entry)))
        name = _POLARS_NAMES.get(name, name)
        if self.at('('):
            return CHType(name, args=tuple(self.items(lambda: self.take())))
        return CHType(name)

    def tuple_field(self) -> tuple[str, CHType]:
        if self.peek()[0] == 'name' and self.peek(1)[0] == 'name':
            return self.take('name'), self.type()
        raise ValueError(f'unnamed Tuple element at token {self.i}')

    def enum_entry(self) -> str:
        label = self.take('str')
        if self.at('='):
            self.take('sym', '=')
            return f'{label} = {self.take("num")}'
        return label


def parse_type(text: str) -> CHType:
    """Parse one ClickHouse type expression."""
    parser = _Parser(text)
    parsed = parser.type()
    if parser.peek()[0] != 'end':
        raise ValueError(f'trailing text after type in {text!r}')
    return parsed


def parse_table(sql: str) -> dict[str, CHType]:
    """The columns of the first `CREATE TABLE ... (...)` in `sql`, in order."""
    sql = re.sub(r'--[^\n]*', '', sql)
    match = _CREATE.search(sql)
    if match is None:
        raise ValueError('no CREATE TABLE found')
    parser = _Parser(sql[match.end() - 1 :])
    columns = parser.items(lambda: (parser.take('name'), parser.type()))
    return dict(columns)


def render(t: CHType) -> str:
    """The type as ClickHouse spells it, normalised."""
    if t.name == 'Array' and t.element is not None:
        text = f'Array({render(t.element)})'
    elif t.name == 'Tuple':
        text = f'Tuple({", ".join(f"{n} {render(f)}" for n, f in t.fields)})'
    elif t.args:
        text = f'{t.name}({", ".join(t.args)})'
    else:
        text = t.name
    if t.low_cardinality:
        text = f'LowCardinality({text})'
    if t.nullable:
        text = f'Nullable({text})'
    return text


# models


def model_type(node: Node) -> CHType:
    """The ClickHouse type derived from a node's polars dtype, ignoring any `ClickHouseType`."""
    if node.kind == 'list':
        assert node.element is not None
        return CHType('Array', element=model_type(node.element))
    if node.kind == 'struct':
        return CHType('Tuple', fields=tuple((f.name, model_type(f.node)) for f in node.fields))
    dtype = node.dtype
    if isinstance(dtype, pl.Enum):
        return CHType('Enum', nullable=node.nullable, args=tuple(f"'{c}'" for c in dtype.categories))
    name = type(dtype).__name__
    return CHType(_POLARS_NAMES.get(name, name), nullable=node.nullable)


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
        if b.name in _ENUMS:
            if allowed_str:
                if set(allowed) != _enum_labels(b):
                    cats.append('type')
            elif allowed:
                if set(allowed) != _enum_values(b):
                    cats.append('type')
            elif a.name not in _ENUMS:
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


@dataclass(frozen=True)
class _Serving:
    script: str
    text: str
    table: str

    def read(self, path: str, *, landed: bool) -> str:
        """`named:` if the field's name is a word in the script, `pass:` if it selects * from the table."""
        if not self.script:
            return 'no'
        leaf = path.rsplit('/', 1)[-1].removesuffix('[]') if path else ''
        if leaf and re.search(rf'\b{re.escape(leaf)}\b', self.text):
            return f'named:{self.script}'
        star = rf'select\s+\*(?:\s+except\s+(?:\w+|\([^)]*\)))?\s+from\s+{re.escape(self.table)}\b'
        if landed and re.search(star, self.text, re.IGNORECASE):
            return f'pass:{self.script}'
        return 'no'


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


def _serving(entry: dict, scripts_dir: Path) -> _Serving:
    script = entry.get('postload_script')
    if not script:
        return _Serving('', '', entry.get('table', ''))
    name = Path(script).name
    text = re.sub(r'--[^\n]*', '', (scripts_dir / name).read_text())
    return _Serving(name, text, entry['table'])


def _models_for(entry: dict, models: dict[str, type[DatasetModel]]) -> list[type[DatasetModel]] | None:
    input_dir = entry.get('input_dir', '')
    if not input_dir.startswith('output'):
        return None
    if 'glob_pattern' in entry:
        pattern = entry['glob_pattern'].split('/')[0]
        return [m for name, m in sorted(models.items()) if fnmatch(name, pattern)] or None
    model = models.get(input_dir.rstrip('/').rsplit('/', 1)[-1])
    return [model] if model else None


def _dataset_rows(dataset: str, entry: dict, schema_dir: Path, scripts_dir: Path,
                  models: dict[str, type[DatasetModel]]) -> Iterator[dict[str, str]]:  # fmt: skip
    serving = _serving(entry, scripts_dir)
    matched = _models_for(entry, models)
    if matched is None:
        yield dict.fromkeys(COLUMNS, '') | {'category': 'no_model', 'dataset': dataset, 'serving_read': 'no'}
        return
    ddl = parse_table((schema_dir / Path(entry['preload_script']).name).read_text())
    union = len(matched) > 1 or 'glob_pattern' in entry
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
    """One row per disagreement between each landing table and its model(s); `no_model` for the rest."""
    models = discover_datasets() if models is None else models
    config = yaml.safe_load(datasets_yaml.read_text())['clickhouse']
    rows = [
        row
        for dataset, entry in config.items()
        if not entry.get('derived')
        for row in _dataset_rows(dataset, entry, schema_dir, scripts_dir, models)
    ]
    return pl.DataFrame(rows, schema=dict.fromkeys(COLUMNS, pl.String))


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit('usage: schema_drift.py <out.csv>')
    pos = Path(__file__).resolve().parents[1]
    clickhouse = pos / 'config' / 'clickhouse'
    frame = drift(pos / 'config' / 'datasets.yaml', clickhouse / 'schema', clickhouse / 'scripts')
    frame.write_csv(sys.argv[1])
    read = pl.col('serving_read').str.extract(r'^(\w+)')
    counts = frame.group_by('category').agg(
        items=pl.len(), **{r: (read == r).sum() for r in ('named', 'pass', 'no')}
    )
    with pl.Config(tbl_rows=-1, tbl_hide_column_data_types=True, tbl_hide_dataframe_shape=True):
        sys.stdout.write(f'{counts.sort("category")}\n')


if __name__ == '__main__':
    main()
