"""ClickHouse landing DDL generated from the `ot_datamodels` dataset models, and a parser for ClickHouse types."""

import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from fnmatch import fnmatch

import polars as pl
from ot_datamodels.base import DatasetModel
from ot_datamodels.discovery import discover_datasets
from ot_datamodels.spec import Node, dataset_spec

_ENUMS = frozenset({'Enum', 'Enum8', 'Enum16'})
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
    cols = parser.items(lambda: (parser.take('name'), parser.type()))
    return dict(cols)


def render(t: CHType) -> str:
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


def _quote(label: str) -> str:
    return "'" + label.replace('\\', '\\\\').replace("'", "\\'") + "'"


def _primitive(node: Node) -> CHType:
    """The non-null type of a primitive: its marker, else its dtype."""
    if node.clickhouse:
        return replace(parse_type(node.clickhouse), nullable=False)
    dtype = node.dtype
    if isinstance(dtype, pl.Enum):
        categories = dtype.categories.to_list()
        if len(categories) > _ENUM8_MAX:
            raise ValueError(f'{len(categories)} Enum categories do not fit Enum8')
        return CHType('Enum8', args=tuple(f'{_quote(c)} = {i}' for i, c in enumerate(categories, 1)))
    name = type(dtype).__name__
    return CHType(_POLARS_NAMES.get(name, name))


def column_type(node: Node, in_nullable_struct: bool = False) -> CHType:
    """The landing type of a node; every field of a nullable struct is Nullable, down to the next Array."""
    if node.kind == 'list':
        assert node.element is not None
        return CHType('Array', element=column_type(node.element))
    if node.kind == 'struct':
        inner = in_nullable_struct or node.nullable
        return CHType('Tuple', fields=tuple((f.name, column_type(f.node, inner)) for f in node.fields))
    return replace(_primitive(node), nullable=node.nullable or in_nullable_struct)


def columns(model: type[DatasetModel]) -> list[tuple[str, CHType]]:
    """The model's top-level columns and their landing types, in model order."""
    return [(f.name, column_type(f.node)) for f in dataset_spec(model).fields]


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
    body = ', '.join(f'`{name}` {render(t)}' for name, t in cols)
    return f'CREATE TABLE IF NOT EXISTS `{table}` ({body}) ENGINE = {engine}'


def members(pattern: str, product: str) -> list[type[DatasetModel]]:
    """The models whose `dataset_name` matches the glob and that are loaded for `product`, by name."""
    models = discover_datasets()
    return [m for name, m in sorted(models.items()) if fnmatch(name, pattern) and product in m.products]
