"""Declare the ClickHouse tables and dictionaries built from a dataset.

A `DatasetModel` lists them in `clickhouse`; pos renders the DDL and loads them. Column names in
`key`, `order_by` and `primary_key` are the table's own, after `columns` and `renames`.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from ot_datamodels.base import DatasetModel


@dataclass(frozen=True)
class ClickHouseTable:
    """A table built from a dataset.

    Attributes:
        name: the table's name.
        engine: `rocksdb` (point lookups on `key`) or `mergetree` (ordered by `order_by`).
        key: the RocksDB key, one column unique in the dataset.
        order_by: the MergeTree sort order.
        primary_key: the MergeTree primary key, a prefix of `order_by`.
        columns: the dataset's columns to keep, in order; all when `None`.
        renames: dataset column to table column.
        where: a row filter, as a ClickHouse expression.
        optimize: run `OPTIMIZE FINAL` after loading.
        products: the products the table is built for; the dataset's own when `None`.
    """

    name: str
    engine: Literal['rocksdb', 'mergetree']
    key: str | None = None
    order_by: tuple[str, ...] = ()
    primary_key: tuple[str, ...] = ()
    columns: tuple[str, ...] | None = None
    renames: Mapping[str, str] = field(default_factory=dict)
    where: str | None = None
    optimize: bool = False
    products: tuple[str, ...] | None = None


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


def _table_columns(table: ClickHouseTable, fields: set[str]) -> list[str]:
    kept = sorted(fields) if table.columns is None else [c for c in table.columns if c in fields]
    return [table.renames.get(c, c) for c in kept]


def validate_declarations(model: 'type[DatasetModel]') -> list[str]:
    """The problems in a model's ClickHouse declarations; empty when they are valid."""
    fields = set(model.model_fields)
    problems: list[str] = []
    tables = {t.name: t for t in model.clickhouse if isinstance(t, ClickHouseTable)}

    for table in tables.values():
        where = f'{model.__name__}.{table.name}'
        if table.engine == 'rocksdb' and table.key is None:
            problems.append(f'{where}: rocksdb needs a key')
        if table.engine == 'mergetree' and not table.order_by:
            problems.append(f'{where}: mergetree needs an order_by')
        problems += [
            f'{where}: {what} {c!r} is not a column of the model'
            for what, cs in (('columns', table.columns or ()), ('renames', table.renames))
            for c in cs
            if c not in fields
        ]
        present = set(_table_columns(table, fields))
        problems += [
            f'{where}: {what} {c!r} is not a column of the table'
            for what, cs in (('key', (table.key,) if table.key else ()), ('order_by', table.order_by))
            for c in cs
            if c not in present
        ]
        if table.primary_key != table.order_by[: len(table.primary_key)]:
            problems.append(f'{where}: primary_key is not a prefix of order_by')

    for dictionary in model.clickhouse:
        if not isinstance(dictionary, ClickHouseDictionary):
            continue
        where = f'{model.__name__}.{dictionary.name}'
        source = tables.get(dictionary.source_table)
        if source is None:
            problems.append(f'{where}: source_table {dictionary.source_table!r} is not declared on the model')
            continue
        known = fields | set(_table_columns(source, fields))
        for c in (dictionary.key, *dictionary.attributes):
            if dictionary.source_columns.get(c, c) not in known:
                problems.append(f'{where}: column {c!r} is not a column of {dictionary.source_table!r}')
    return problems
