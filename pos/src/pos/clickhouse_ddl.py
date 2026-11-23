"""ClickHouse landing DDL generated from the `ot_datamodels` dataset models, and a parser for ClickHouse types."""

import argparse
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from fnmatch import fnmatch
from pathlib import Path

import polars as pl
import yaml
from ot_datamodels.base import DatasetModel
from ot_datamodels.clickhouse import ClickHouseDictionary, ClickHouseTable
from ot_datamodels.discovery import discover_models
from ot_datamodels.spec import Node, dataset_spec

ENUMS = frozenset({'Enum', 'Enum8', 'Enum16'})
_POLARS_NAMES = {'Boolean': 'Bool'}
_ENUM8_MAX = 127
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


# parsing


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
        if name in ENUMS:
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
    cols = parser.items(lambda: (parser.take('name'), parser.type()))
    return dict(cols)


def render(t: CHType, *, nullable_outermost: bool = False, quote_fields: bool = True) -> str:
    """The type as ClickHouse accepts it: `LowCardinality(Nullable(...))`, Tuple field names backtick-quoted.

    `nullable_outermost` gives `Nullable(LowCardinality(...))` instead; `quote_fields=False` leaves field names bare.
    """
    sub = lambda u: render(u, nullable_outermost=nullable_outermost, quote_fields=quote_fields)
    if t.name == 'Array' and t.element is not None:
        text = f'Array({sub(t.element)})'
    elif t.name == 'Tuple':
        text = f'Tuple({", ".join(f"{_name(n) if quote_fields else n} {sub(f)}" for n, f in t.fields)})'
    elif t.args:
        text = f'{t.name}({", ".join(t.args)})'
    else:
        text = t.name
    wrappers = {'Nullable': t.nullable, 'LowCardinality': t.low_cardinality}
    order = ('LowCardinality', 'Nullable') if nullable_outermost else ('Nullable', 'LowCardinality')
    for wrapper in order:
        if wrappers[wrapper]:
            text = f'{wrapper}({text})'
    return text


def _name(name: str) -> str:
    return f'`{name}`'


def canonical(t: CHType) -> CHType:
    """`t` with every Enum an `Enum8`/`Enum16` with explicit values; an entry with none takes the last value + 1."""
    element = canonical(t.element) if t.element is not None else None
    fields = tuple((n, canonical(f)) for n, f in t.fields)
    if t.name not in ENUMS:
        return replace(t, element=element, fields=fields)
    entries, value = [], 0
    for entry in t.args:
        label, _, given = entry.partition(' = ')
        value = int(given) if given else value + 1
        entries.append(f'{label} = {value}')
    name = t.name
    if name == 'Enum':
        name = 'Enum8' if all(-128 <= int(e.rsplit(' = ', 1)[1]) <= _ENUM8_MAX for e in entries) else 'Enum16'
    return replace(t, name=name, args=tuple(entries))


# models


def _quote(label: str) -> str:
    return "'" + label.replace('\\', '\\\\').replace("'", "\\'") + "'"


def model_type(node: Node, path: str = '') -> CHType:
    """The ClickHouse type derived from a node's polars dtype, ignoring any `ClickHouseType`."""
    if node.kind == 'list':
        assert node.element is not None
        return CHType('Array', element=model_type(node.element, f'{path}[]'))
    if node.kind == 'struct':
        return CHType('Tuple', fields=tuple((f.name, model_type(f.node, f'{path}/{f.name}')) for f in node.fields))
    return replace(_dtype_type(node, path), nullable=node.nullable)


def _primitive(node: Node, path: str) -> CHType:
    """The non-null type of a primitive: its marker, else its dtype."""
    if node.clickhouse:
        return replace(parse_type(node.clickhouse), nullable=False)
    return _dtype_type(node, path)


def _dtype_type(node: Node, path: str) -> CHType:
    """The non-null type of a primitive's polars dtype."""
    dtype = node.dtype
    if isinstance(dtype, pl.Enum):
        categories = dtype.categories.to_list()
        if len(categories) > _ENUM8_MAX:
            raise ValueError(f'{path or "value"}: {len(categories)} Enum categories do not fit Enum8')
        return CHType('Enum8', args=tuple(f'{_quote(c)} = {i}' for i, c in enumerate(categories, 1)))
    name = type(dtype).__name__
    return CHType(_POLARS_NAMES.get(name, name))


def _type(node: Node, force: bool, path: str) -> CHType:
    if node.kind == 'list':
        assert node.element is not None
        return CHType('Array', element=_type(node.element, False, f'{path}[]'))
    if node.kind == 'struct':
        inner = force or node.nullable
        return CHType('Tuple', fields=tuple((f.name, _type(f.node, inner, f'{path}/{f.name}')) for f in node.fields))
    return replace(_primitive(node, path), nullable=node.nullable or force)


def column_type(node: Node, in_nullable_struct: bool = False) -> CHType:
    """The landing type of a node; every field of a nullable struct is Nullable, down to the next Array."""
    return _type(node, in_nullable_struct, '')


def columns(model: type[DatasetModel]) -> list[tuple[str, CHType]]:
    """The model's top-level columns and their landing types, in model order."""
    return [(f.name, _type(f.node, False, f.name)) for f in dataset_spec(model).fields]


def _non_null(t: CHType) -> CHType:
    element = _non_null(t.element) if t.element is not None else None
    fields = tuple((n, _non_null(f)) for n, f in t.fields)
    return replace(t, nullable=False, element=element, fields=fields)


def _absent(t: CHType) -> CHType:
    """The type of a column a member lacks: Nullable, all fields Nullable for a Tuple, unchanged for an Array."""
    if t.name == 'Array':
        return t
    if t.name == 'Tuple':
        return replace(t, fields=tuple((n, _absent(f)) for n, f in t.fields))
    return replace(t, nullable=True)


def _either_nullable(a: CHType, b: CHType) -> CHType:
    """`a` (equal to `b` but for nullability), Nullable wherever either is."""
    element = _either_nullable(a.element, b.element) if a.element is not None and b.element is not None else None
    fields = tuple((n, _either_nullable(f, g)) for (n, f), (_, g) in zip(a.fields, b.fields, strict=True))
    return replace(a, nullable=a.nullable or b.nullable, element=element, fields=fields)


def union_columns(models: Sequence[type[DatasetModel]]) -> list[tuple[str, CHType]]:
    """Every column of every model, first-seen order; Nullable where any model has it nullable or lacks it."""
    per_model = [(m.dataset_name, dict(columns(m))) for m in models]
    first: dict[str, tuple[str, CHType]] = {}
    for name, cols in per_model:
        for col, t in cols.items():
            if col not in first:
                first[col] = (name, t)
                continue
            seen_in, seen = first[col]
            if _non_null(seen) != _non_null(t):
                raise ValueError(f'column {col}: {seen_in} has {render(seen)}, {name} has {render(t)}')
    out = []
    for col, (_, t) in first.items():
        for _, cols in per_model:
            t = _either_nullable(t, cols.get(col) or _absent(t))
        out.append((col, t))
    return out


def create_table(table: str, cols: Sequence[tuple[str, CHType]], engine: str = 'Log') -> str:
    """`CREATE TABLE IF NOT EXISTS` for the columns, identifiers backtick-quoted."""
    body = ', '.join(f'{_name(name)} {render(t)}' for name, t in cols)
    return f'CREATE TABLE IF NOT EXISTS {_name(table)} ({body}) ENGINE = {engine}'


def declared_tables(model: type[DatasetModel], product: str) -> list[ClickHouseTable]:
    """The tables the model declares for `product`; a table with no `products` follows the model's."""
    return [
        t for t in model.clickhouse if isinstance(t, ClickHouseTable) and product in (t.products or model.products)
    ]


def _kept(table: ClickHouseTable, cols: Sequence[tuple[str, CHType]]) -> list[str]:
    return [name for name, _ in cols] if table.columns is None else list(table.columns)


def table_columns(table: ClickHouseTable, cols: Sequence[tuple[str, CHType]]) -> list[tuple[str, CHType]]:
    """The table's columns from the dataset's `cols`: those in `columns` (all when None), renamed, in order."""
    by_name = dict(cols)
    return [(table.renames.get(name, name), by_name[name]) for name in _kept(table, cols)]


def table_sql(table: ClickHouseTable, cols: Sequence[tuple[str, CHType]], landing: str) -> list[str]:
    """The statements that create and fill a declared table from its landing table; `cols` are the dataset's."""
    own = table_columns(table, cols)
    types = dict(own)
    if table.engine == 'rocksdb':
        engine = f'EmbeddedRocksDB() PRIMARY KEY {_name(table.key or "")}'
    else:
        engine = f'MergeTree() ORDER BY ({_names(table.order_by)})'
        if table.primary_key:
            engine += f' PRIMARY KEY ({_names(table.primary_key)})'
        if any(types[c].nullable for c in table.order_by):
            engine += ' SETTINGS allow_nullable_key = 1'
    select = ', '.join(
        f'{_name(c)} AS {_name(table.renames[c])}' if c in table.renames else _name(c) for c in _kept(table, cols)
    )
    insert = f'INSERT INTO {_name(table.name)} SELECT {select} FROM {_name(landing)}'  # noqa: S608 quoted identifiers
    if table.where:
        insert += f' WHERE {table.where}'
    statements = [create_table(table.name, own, engine), insert]
    if table.optimize:
        statements.append(f'OPTIMIZE TABLE {_name(table.name)} FINAL')
    return statements


def dictionary_sql(d: ClickHouseDictionary, cols: Sequence[tuple[str, CHType]]) -> str:
    """`CREATE DICTIONARY` over its source table; `cols` are the table's own. Renames go through EXPRESSION."""
    types = dict(cols)
    parts = []
    for name in (d.key, *d.attributes):
        source = d.source_columns.get(name, name)
        expression = f' EXPRESSION {_name(source)}' if source != name else ''
        parts.append(f'{_name(name)} {render(types[source])}{expression}')
    return (
        f'CREATE DICTIONARY IF NOT EXISTS {_name(d.name)} ({", ".join(parts)}) PRIMARY KEY {_name(d.key)} '
        f"SOURCE(CLICKHOUSE(TABLE '{d.source_table}')) LAYOUT(COMPLEX_KEY_HASHED()) LIFETIME(0)"
    )


def uniqueness_sql(table: ClickHouseTable) -> str:
    """A query that throws when the table's RocksDB key repeats."""
    key = table.key
    return f"SELECT throwIf(count() != uniqExact({key}), 'duplicate {key} in {table.name}') FROM {table.name}"  # noqa: S608 declared names


def _names(names: Sequence[str]) -> str:
    return ', '.join(_name(n) for n in names)


def members(pattern: str, product: str) -> list[type[DatasetModel]]:
    """The models whose `dataset_name` matches the glob and that are loaded for `product`, by name."""
    models = discover_models()
    return [m for name, m in sorted(models.items()) if fnmatch(name, pattern) and product in m.products]


def landing_columns(entry: Mapping, product: str) -> list[tuple[str, CHType]] | None:
    """The generated columns of a `datasets.yaml` clickhouse entry; None if it has no model.

    A `models` entry is the union of its members for `product`; a `model` entry is that model, whatever its
    products (a product without the output loads an empty table).
    """
    if 'models' in entry:
        return union_columns(members(entry['models'], product))
    if 'model' in entry:
        return columns(discover_models()[entry['model']])
    return None


def member_table(model: type[DatasetModel]) -> str:
    """The landing table of one member of a `models` entry."""
    return f'{model.dataset_name}_log'


def ddl_for(entry: Mapping, product: str) -> list[tuple[str, str]]:
    """(table, CREATE TABLE) pairs a load of the entry runs: its one table, or each member then the union."""
    cols = landing_columns(entry, product)
    if cols is None:
        return []
    union = (entry['table'], create_table(entry['table'], cols, entry.get('engine', 'Log')))
    if 'models' not in entry:
        return [union]
    found = members(entry['models'], product)
    if not found:
        raise ValueError(f'{entry["table"]}: models {entry["models"]!r} matches no model for {product}')
    return [(member_table(m), create_table(member_table(m), columns(m))) for m in found] + [union]


def main() -> None:
    parser = argparse.ArgumentParser(description='Print the generated ClickHouse landing DDL.')
    parser.add_argument('dataset', nargs='?', help='a clickhouse entry of the datasets config')
    parser.add_argument('--all', action='store_true', help='every entry with a model')
    parser.add_argument('--product', default='platform', choices=('platform', 'ppp'))
    parser.add_argument('--config', default='config/datasets.yaml', help='the datasets config (default: %(default)s)')
    args = parser.parse_args()
    if bool(args.dataset) == args.all:
        parser.error('give one dataset or --all')
    entries = yaml.safe_load(Path(args.config).read_text())['clickhouse']
    if args.dataset and args.dataset not in entries:
        parser.error(f'no clickhouse entry {args.dataset}')
    names = list(entries) if args.all else [args.dataset]
    statements = [sql for name in names for _, sql in ddl_for(entries[name], args.product)]
    if not statements:
        parser.error(f'{args.dataset} has no model: its DDL is {entries[args.dataset].get("preload_script")}')
    sys.stdout.write(''.join(f'{sql};\n' for sql in statements))


def type_at(cols: Sequence[tuple[str, CHType]], path: str) -> CHType | None:
    """The type at `a`, `a/b` or `a[]/b` (`a[]` is the element); a Tuple of `cols` for ''; None if absent."""
    t: CHType | None = CHType('Tuple', fields=tuple(cols))
    for part in path.split('/') if path else []:
        name, lists = part, 0
        while name.endswith('[]'):
            name, lists = name[:-2], lists + 1
        t = dict(t.fields).get(name) if t is not None else None
        for _ in range(lists):
            t = t.element if t is not None else None
    return t
