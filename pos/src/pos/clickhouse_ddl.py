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
from ot_datamodels.discovery import discover_models
from ot_datamodels.spec import Node, dataset_spec

from pos.clickhouse_tables import (
    ClickHouseDictionary,
    ClickHouseTable,
    dictionary_problems,
    entry_dictionaries,
    entry_tables,
    table_problems,
)

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


def create_table(table: str, cols: Sequence[tuple[str, CHType]], engine: str = 'Log', *, replace: bool = False) -> str:
    """`CREATE TABLE IF NOT EXISTS` for the columns, identifiers backtick-quoted; plain `CREATE TABLE` for `replace`."""
    body = ', '.join(f'{_name(name)} {render(t)}' for name, t in cols)
    exists = '' if replace else 'IF NOT EXISTS '
    return f'CREATE TABLE {exists}{_name(table)} ({body}) ENGINE = {engine}'


def _primary_keys(models: Sequence[type[DatasetModel]]) -> dict[str, tuple[str, ...]]:
    return {m.dataset_name: dataset_spec(m).primary_key for m in models}


def _disagreement(keys: Mapping[str, tuple[str, ...]]) -> str:
    return 'members disagree on PrimaryKey: ' + ', '.join(f'{name} {key}' for name, key in sorted(keys.items()))


def rocksdb_key(models: Sequence[type[DatasetModel]]) -> str:
    """The RocksDB key of a table built from `models`: the one `PrimaryKey` column they all have."""
    keys = _primary_keys(models)
    shared = set(keys.values())
    if len(shared) > 1:
        raise ValueError(_disagreement(keys))
    (key,) = shared or {()}
    if len(key) != 1:
        raise ValueError(f'{", ".join(sorted(keys))}: rocksdb needs one PrimaryKey column, found {key or "none"}')
    return key[0]


def table_columns(table: ClickHouseTable, cols: Sequence[tuple[str, CHType]]) -> list[tuple[str, CHType]]:
    """The table's columns from the dataset's `cols`: those in `columns` (all when None), in order."""
    by_name = dict(cols)
    return list(cols) if table.columns is None else [(name, by_name[name]) for name in table.columns]


def table_sql(
    table: ClickHouseTable, cols: Sequence[tuple[str, CHType]], landing: str, key: str | None = None
) -> list[str]:
    """The statements that rebuild a declared table from its landing table; `cols` are the dataset's.

    A RocksDB table is keyed on `key`. A MergeTree table's primary key is its sort key; it allows a nullable key
    when an `order_by` column is nullable and is optimised after the insert. The statements drop the table first,
    so a re-run leaves one copy of the rows.
    """
    own = table_columns(table, cols)
    types = dict(own)
    if table.engine == 'rocksdb':
        if key is None:
            raise ValueError(f'{table.name}: rocksdb needs a key')
        engine = f'EmbeddedRocksDB() PRIMARY KEY {_name(key)}'
    else:
        engine = f'MergeTree() ORDER BY ({_names(table.order_by)})'
        if any(types[c].nullable for c in table.order_by):
            engine += ' SETTINGS allow_nullable_key = 1'
    select = _names([name for name, _ in own])
    insert = f'INSERT INTO {_name(table.name)} SELECT {select} FROM {_name(landing)}'  # noqa: S608 quoted identifiers
    if table.where:
        insert += f' WHERE {table.where}'
    statements = [
        f'DROP TABLE IF EXISTS {_name(table.name)} SYNC',
        create_table(table.name, own, engine, replace=True),
        insert,
    ]
    if table.engine == 'mergetree':
        statements.append(f'OPTIMIZE TABLE {_name(table.name)} FINAL')
    return statements


def dictionary_sql(d: ClickHouseDictionary, cols: Sequence[tuple[str, CHType]]) -> str:
    """`CREATE DICTIONARY` over its source table; `cols` are the table's own. Renames go through EXPRESSION.

    Columns take their source's type without LowCardinality, which dictionaries do not accept.
    """
    types = dict(cols)
    parts = []
    for name in (d.key, *d.attributes):
        source = d.source_columns.get(name, name)
        expression = f' EXPRESSION {_name(source)}' if source != name else ''
        parts.append(f'{_name(name)} {render(replace(types[source], low_cardinality=False))}{expression}')
    return (
        f'CREATE OR REPLACE DICTIONARY {_name(d.name)} ({", ".join(parts)}) PRIMARY KEY {_name(d.key)} '
        f"SOURCE(CLICKHOUSE(TABLE '{d.source}')) LAYOUT(COMPLEX_KEY_HASHED()) LIFETIME(0)"
    )


def drop_dictionary_sql(d: ClickHouseDictionary) -> str:
    """Drops the dictionary; run it before `table_sql` on the source table, which ClickHouse will not drop under it."""
    return f'DROP DICTIONARY IF EXISTS {_name(d.name)} SYNC'


def uniqueness_sql(table: ClickHouseTable, key: str, landing: str) -> str:
    """A query that throws when the RocksDB `key` repeats in the rows the table takes from `landing`.

    It reads the landing table: RocksDB keeps the last row of a repeated key, so the table itself never repeats one.
    A NULL key counts as a value.
    """
    where = f' WHERE {table.where}' if table.where else ''
    return (
        f"SELECT throwIf(count() != uniqExact(tuple({_name(key)})), 'duplicate {key} in {table.name}') "  # noqa: S608 declared names
        f'FROM {_name(landing)}{where}'
    )


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
    if (model := entry_model(entry)) is not None:
        return columns(model)
    return None


def entry_model(entry: Mapping) -> type[DatasetModel] | None:
    """The model of a `model` entry of `datasets.yaml`; None for any other entry."""
    return discover_models()[entry['model']] if 'model' in entry else None


def _entry_models(entry: Mapping, product: str) -> list[type[DatasetModel]]:
    """The models an entry loads for `product`: its members, or its one model."""
    model = entry_model(entry)
    return [model] if model is not None else members(entry['models'], product)


def entry_key(entry: Mapping, product: str) -> str:
    """The RocksDB key of an entry's tables: its model's one `PrimaryKey` column, or the one its members share."""
    try:
        return rocksdb_key(_entry_models(entry, product))
    except ValueError as e:
        raise ValueError(f'{entry["table"]}: {e}') from e


def entry_problems(entry: Mapping, product: str) -> list[str]:
    """The problems in the tables and dictionaries a `datasets.yaml` entry declares, checked against its members."""
    tables = entry_tables(entry)
    owner = entry['table']
    keys = _primary_keys(_entry_models(entry, product))
    shared = set(keys.values())
    keyed = any(t.engine == 'rocksdb' for t in tables)
    problems = []
    if tables and 'postload_script' in entry:
        problems.append(f'{owner}: declares ClickHouse tables and has a postload_script')
    problems +=[f'{owner}: {_disagreement(keys)}'] if keyed and len(shared) > 1 else []
    key = next(iter(shared)) if len(shared) == 1 else ()
    fields = [name for name, _ in landing_columns(entry, product) or []]
    problems += [p for t in tables for p in table_problems(t, owner, fields, key)]
    return problems + [p for d in entry_dictionaries(entry) for p in dictionary_problems(d, owner, tables, fields)]


def is_declared(entry: Mapping) -> bool:
    """Whether the entry declares tables under `tables`."""
    return bool(entry.get('tables'))


def declarations(entry: Mapping, product: str) -> tuple[list[ClickHouseTable], list[ClickHouseDictionary]]:
    """The tables and dictionaries an entry declares, for `product`.

    A `model` entry builds them only for its model's products; a `models` entry for any product.
    """
    model = entry_model(entry)
    if model is not None and product not in model.products:
        return [], []
    return entry_tables(entry), entry_dictionaries(entry)


def declared_sql(entry: Mapping, product: str) -> list[str]:
    """The statements that build an entry's declared tables and dictionaries from its filled landing table.

    They end by dropping the landing table. An undeclared entry has none.
    """
    tables, dictionaries = declarations(entry, product)
    if not tables:
        return []
    cols = landing_columns(entry, product) or []
    landing = entry['table']
    key = entry_key(entry, product) if any(t.engine == 'rocksdb' for t in tables) else None
    by_name = {t.name: t for t in tables}
    statements = [drop_dictionary_sql(d) for d in dictionaries]
    for table in tables:
        statements += table_sql(table, cols, landing, key)
    statements += [dictionary_sql(d, table_columns(by_name[d.source], cols)) for d in dictionaries]
    return [*statements, f'DROP TABLE {_name(landing)} SYNC']


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
