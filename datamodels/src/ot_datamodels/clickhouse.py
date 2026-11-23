"""Declare the ClickHouse tables and dictionaries built from a dataset.

A `DatasetModel` lists them in `clickhouse`; pos renders the DDL and loads them. A table states only what the model
cannot: the RocksDB key is the model's `PrimaryKey`, the products are the model's, and a MergeTree table is always
optimised, with `allow_nullable_key` set when an `order_by` column is nullable.
"""

import re
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from ot_datamodels.base import DatasetModel


@dataclass(frozen=True)
class ClickHouseTable:
    """A table built from a dataset.

    Attributes:
        name: the table's name.
        engine: `rocksdb` (point lookups on the model's `PrimaryKey`) or `mergetree` (ordered by `order_by`).
        order_by: the MergeTree sort order, also its primary key; MergeTree only.
        columns: the dataset's top-level columns to keep, in order; all when `None`.
        where: a row filter, as a ClickHouse expression over the dataset's columns.
    """

    name: str
    engine: Literal['rocksdb', 'mergetree']
    order_by: tuple[str, ...] = ()
    columns: tuple[str, ...] | None = None
    where: str | None = None


@dataclass(frozen=True)
class ClickHouseDictionary:
    """A dictionary over a table declared on the same model.

    Attributes:
        name: the dictionary's name.
        source_table: the `ClickHouseTable.name` it reads.
        key: the lookup key column.
        attributes: the dictionary's value columns.
        source_columns: dictionary column to source column, where they differ.
    """

    name: str
    source_table: str
    key: str
    attributes: tuple[str, ...]
    source_columns: Mapping[str, str] = field(default_factory=dict)


_WHERE_TOKEN = re.compile(r"'(?:[^'\\]|\\.)*'|`([^`]*)`|([A-Za-z_]\w*)(\s*\()?")
_KEYWORDS = frozenset({'and', 'or', 'not', 'is', 'null', 'in', 'like', 'ilike', 'between', 'true', 'false'})


def where_columns(where: str) -> set[str]:
    """The column names a `where` expression reads, best effort: identifiers that are not keywords or functions."""
    names = set()
    for m in _WHERE_TOKEN.finditer(where):
        quoted, ident, call = m.groups()
        if quoted is not None:
            names.add(quoted)
        elif ident is not None and not call and ident.lower() not in _KEYWORDS:
            names.add(ident)
    return names


def table_columns(table: ClickHouseTable, fields: Collection[str]) -> list[str]:
    """The table's columns: those in `columns` that are among `fields`, else all `fields`."""
    return list(fields) if table.columns is None else [c for c in table.columns if c in fields]


def table_problems(table: ClickHouseTable, owner: str, fields: Collection[str], key: tuple[str, ...]) -> list[str]:
    """The problems in one table declared on a dataset with columns `fields` and `PrimaryKey` columns `key`."""
    where = f'{owner}.{table.name}'
    problems: list[str] = []
    if table.engine == 'rocksdb':
        if len(key) != 1:
            problems.append(f'{where}: rocksdb needs one PrimaryKey column on the model, it has {key or "none"}')
        if table.order_by:
            problems.append(f'{where}: rocksdb takes no order_by')
    if table.engine == 'mergetree' and not table.order_by:
        problems.append(f'{where}: mergetree needs an order_by')
    problems += [f'{where}: columns {c!r} is not a column of the model' for c in table.columns or () if c not in fields]
    if table.where:
        problems += [
            f'{where}: where {c!r} is not a column of the model'
            for c in sorted(where_columns(table.where))
            if c not in fields
        ]
    present = set(table_columns(table, fields))
    used = [('key', c) for c in key if table.engine == 'rocksdb' and len(key) == 1]
    used += [('order_by', c) for c in table.order_by]
    problems += [f'{where}: {what} {c!r} is not a column of the table' for what, c in used if c not in present]
    return problems


def validate_declarations(model: 'type[DatasetModel]') -> list[str]:
    """The problems in a model's ClickHouse declarations; empty when they are valid."""
    from ot_datamodels.base import PrimaryKey

    fields = list(model.model_fields)
    key = tuple(n for n, f in model.model_fields.items() if any(isinstance(m, PrimaryKey) for m in f.metadata))
    tables = {t.name: t for t in model.clickhouse if isinstance(t, ClickHouseTable)}
    problems = [p for t in tables.values() for p in table_problems(t, model.__name__, fields, key)]

    for dictionary in model.clickhouse:
        if not isinstance(dictionary, ClickHouseDictionary):
            continue
        where = f'{model.__name__}.{dictionary.name}'
        source = tables.get(dictionary.source_table)
        if source is None:
            problems.append(f'{where}: source_table {dictionary.source_table!r} is not declared on the model')
            continue
        known = set(table_columns(source, fields))
        problems += [
            f'{where}: column {c!r} is not a column of {dictionary.source_table!r}'
            for c in (dictionary.key, *dictionary.attributes)
            if dictionary.source_columns.get(c, c) not in known
        ]
    return problems
