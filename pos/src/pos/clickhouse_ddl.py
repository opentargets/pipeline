"""The served views' ClickHouse SQL, generated from their `ot_datamodels` models, and a parser for ClickHouse types.

A view model with `clickhouse = Serving(...)` is one table named after its `dataset_name`, built from the whole view:
it lands in `<dataset_name>_log` from `view/<dataset_name>`, its table and dictionaries are built from the landing
table, which is then dropped.
"""

import argparse
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import polars as pl
from ot_datamodels.base import DatasetModel
from ot_datamodels.discovery import discover_datasets, discover_models
from ot_datamodels.serving import ClickHouseDictionary, Serving
from ot_datamodels.spec import Node, dataset_spec, walk

ENUMS = frozenset({'Enum', 'Enum8', 'Enum16'})
_POLARS_NAMES = {'Boolean': 'Bool'}
_ENUM8_MAX = 127
_TOKEN = re.compile(r"\s*(?:`([^`]*)`|('(?:[^'\\]|\\.)*')|(-?\d+(?:\.\d+)?)|([A-Za-z_]\w*)|(\S))")


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


def _primitive(node: Node, path: str, types: Mapping[str, str]) -> CHType:
    """The non-null type of a primitive: its entry in `types` (a view's `column_types`), else its marker or dtype."""
    if path in types:
        return replace(parse_type(types[path]), nullable=False)
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


def _type(node: Node, force: bool, path: str, types: Mapping[str, str]) -> CHType:
    if node.kind == 'list':
        assert node.element is not None
        return CHType('Array', element=_type(node.element, False, f'{path}[]', types))
    if node.kind == 'struct':
        inner = force or node.nullable
        return CHType(
            'Tuple', fields=tuple((f.name, _type(f.node, inner, f'{path}/{f.name}', types)) for f in node.fields)
        )
    return replace(_primitive(node, path, types), nullable=node.nullable or force)


def column_type(node: Node, in_nullable_struct: bool = False) -> CHType:
    """The landing type of a node; every field of a nullable struct is Nullable, down to the next Array."""
    return _type(node, in_nullable_struct, '', {})


def columns(model: type[DatasetModel]) -> list[tuple[str, CHType]]:
    """The model's top-level columns and their landing types, in model order; a served view's take its `column_types`.

    Raises:
        ValueError: if a `column_types` path is not a primitive value of the view.
    """
    spec = dataset_spec(model)
    types = model.clickhouse.column_types if model.clickhouse is not None else {}
    primitives = {path for path, node in walk(spec) if node.kind == 'primitive'}
    if unknown := sorted(set(types) - primitives):
        raise ValueError(f'{model.dataset_name}: column_types paths {unknown} are not primitive values')
    return [(f.name, _type(f.node, False, f.name, types)) for f in spec.fields]


def create_table(table: str, cols: Sequence[tuple[str, CHType]], engine: str = 'Log', *, replace: bool = False) -> str:
    """`CREATE TABLE IF NOT EXISTS` for the columns, identifiers backtick-quoted; plain `CREATE TABLE` for `replace`."""
    body = ', '.join(f'{_name(name)} {render(t)}' for name, t in cols)
    exists = '' if replace else 'IF NOT EXISTS '
    return f'CREATE TABLE {exists}{_name(table)} ({body}) ENGINE = {engine}'


# served views


def served_views(product: str | None = None) -> dict[str, type[DatasetModel]]:
    """Every served view model (`clickhouse` set), by `dataset_name`; for `product`, only the views it loads.

    Raises:
        ValueError: if an output model declares `clickhouse`.
    """
    if outputs := sorted(name for name, model in discover_datasets().items() if model.clickhouse is not None):
        raise ValueError(f'output models {outputs} declare clickhouse: only a view is served')
    return {
        name: model
        for name, model in sorted(discover_models().items())
        if model.clickhouse is not None and (product is None or product in model.products)
    }


def _serving(model: type[DatasetModel]) -> Serving:
    if model.clickhouse is None:
        raise ValueError(f'{model.dataset_name} is not a served view')
    return model.clickhouse


def landing_table(model: type[DatasetModel]) -> str:
    """The landing table a view loads into: `<dataset_name>_log`."""
    return f'{model.dataset_name}_log'


def input_dir(model: type[DatasetModel]) -> str:
    """The folder a view loads from, under the synced release data: `view/<dataset_name>`."""
    return f'view/{model.dataset_name}'


def landing_ddl(model: type[DatasetModel]) -> str:
    """`CREATE TABLE` of the view's landing table, a Log table of its columns."""
    return create_table(landing_table(model), columns(model))


def rocksdb_key(model: type[DatasetModel]) -> str:
    """The RocksDB key of a view's table: the model's one `PrimaryKey` column."""
    key = dataset_spec(model).primary_key
    if len(key) != 1:
        raise ValueError(f'{model.dataset_name}: rocksdb needs one PrimaryKey column, found {key or "none"}')
    return key[0]


def table_sql(model: type[DatasetModel]) -> list[str]:
    """The statements that rebuild a view's table, named `dataset_name`, from its landing table.

    A RocksDB table is keyed on the model's `PrimaryKey`. A MergeTree table's primary key is its sort key; it allows
    a nullable key when an `order_by` column is nullable and is optimised after the insert. The statements drop the
    table first, so a re-run leaves one copy of the rows.
    """
    serving, table, cols = _serving(model), model.dataset_name, columns(model)
    if serving.engine == 'rocksdb':
        engine = f'EmbeddedRocksDB() PRIMARY KEY {_name(rocksdb_key(model))}'
    else:
        engine = f'MergeTree() ORDER BY ({_names(serving.order_by)})'
        if any(t.nullable for name, t in cols if name in serving.order_by):
            engine += ' SETTINGS allow_nullable_key = 1'
    select = _names([name for name, _ in cols])
    statements = [
        f'DROP TABLE IF EXISTS {_name(table)} SYNC',
        create_table(table, cols, engine, replace=True),
        f'INSERT INTO {_name(table)} SELECT {select} FROM {_name(landing_table(model))}',  # noqa: S608 quoted identifiers
    ]
    if serving.engine == 'mergetree':
        statements.append(f'OPTIMIZE TABLE {_name(table)} FINAL')
    return statements


def dictionary_sql(d: ClickHouseDictionary, table: str, cols: Sequence[tuple[str, CHType]]) -> str:
    """`CREATE DICTIONARY` over the table `table`; `cols` are the table's own. Renames go through EXPRESSION.

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
        f"SOURCE(CLICKHOUSE(TABLE '{table}')) LAYOUT(COMPLEX_KEY_HASHED()) LIFETIME(0)"
    )


def drop_dictionary_sql(d: ClickHouseDictionary) -> str:
    """Drops the dictionary; run it before `table_sql` on the source table, which ClickHouse will not drop under it."""
    return f'DROP DICTIONARY IF EXISTS {_name(d.name)} SYNC'


def served_sql(model: type[DatasetModel]) -> list[str]:
    """The statements that build a view's table and dictionaries from its filled landing table, then drop it."""
    dictionaries, cols = _serving(model).dictionaries, columns(model)
    return [
        *(drop_dictionary_sql(d) for d in dictionaries),
        *table_sql(model),
        *(dictionary_sql(d, model.dataset_name, cols) for d in dictionaries),
        f'DROP TABLE {_name(landing_table(model))} SYNC',
    ]


def uniqueness_sql(model: type[DatasetModel]) -> str:
    """A query that throws when the RocksDB key repeats in the view's landing table.

    It reads the landing table: RocksDB keeps the last row of a repeated key, so the table itself never repeats one.
    A NULL key counts as a value.
    """
    key, table = rocksdb_key(model), model.dataset_name
    return (
        f"SELECT throwIf(count() != uniqExact(tuple({_name(key)})), 'duplicate {key} in {table}') "  # noqa: S608 declared names
        f'FROM {_name(landing_table(model))}'
    )


def _names(names: Sequence[str]) -> str:
    return ', '.join(_name(n) for n in names)


def main() -> None:
    parser = argparse.ArgumentParser(description='Print the generated ClickHouse landing DDL of the served views.')
    parser.add_argument('dataset', nargs='?', help='a served view')
    parser.add_argument('--all', action='store_true', help='every served view of the product')
    parser.add_argument('--product', default='platform', choices=('platform', 'ppp'))
    args = parser.parse_args()
    if bool(args.dataset) == args.all:
        parser.error('give one dataset or --all')
    views = served_views()
    if args.dataset and args.dataset not in views:
        parser.error(f'no served view {args.dataset}')
    models = list(served_views(args.product).values()) if args.all else [views[args.dataset]]
    sys.stdout.write(''.join(f'{landing_ddl(model)};\n' for model in models))
