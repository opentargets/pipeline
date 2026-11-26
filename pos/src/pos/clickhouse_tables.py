"""The ClickHouse tables and dictionaries an entry declares under `tables` and `dictionaries`.

An entry is a `datasets.yaml` one, or one derived from a served view model (`pos.clickhouse_ddl.served_entries`).

A table states only what the dataset's model cannot: the RocksDB key is the model's `PrimaryKey`, the products are
the model's, and a MergeTree table is always optimised, with `allow_nullable_key` set when an `order_by` column is
nullable.
"""

import re
from collections.abc import Collection, Mapping
from dataclasses import dataclass, fields
from typing import Literal

from ot_datamodels.serving import ClickHouseDictionary


@dataclass(frozen=True)
class ClickHouseTable:
    """A table built from a dataset's landing table.

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


def _known(entry: Mapping, kind: type[ClickHouseTable] | type[ClickHouseDictionary], item: Mapping) -> None:
    if unknown := set(item) - {f.name for f in fields(kind)}:
        raise ValueError(f'{entry["table"]}: unknown {kind.__name__} fields {sorted(unknown)}')


def entry_tables(entry: Mapping) -> list[ClickHouseTable]:
    """The tables listed under `tables` in a `datasets.yaml` entry."""
    tables = []
    for t in entry.get('tables', []):
        _known(entry, ClickHouseTable, t)
        tables.append(
            ClickHouseTable(
                name=t['name'],
                engine=t['engine'],
                order_by=tuple(t.get('order_by', ())),
                columns=tuple(t['columns']) if 'columns' in t else None,
                where=t.get('where'),
            )
        )
    return tables


def entry_dictionaries(entry: Mapping) -> list[ClickHouseDictionary]:
    """The dictionaries listed under `dictionaries` in a `datasets.yaml` entry."""
    dictionaries = []
    for d in entry.get('dictionaries', []):
        _known(entry, ClickHouseDictionary, d)
        dictionaries.append(
            ClickHouseDictionary(
                name=d['name'],
                source=d['source'],
                key=d['key'],
                attributes=tuple(d['attributes']),
                source_columns=dict(d.get('source_columns', {})),
            )
        )
    return dictionaries


def source_table(dictionary: ClickHouseDictionary) -> str:
    """The table a dictionary reads; an entry's dictionaries always name it (`served_entries` fills it in)."""
    if dictionary.source is None:
        raise ValueError(f'{dictionary.name}: no source table')
    return dictionary.source


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


def column_names(table: ClickHouseTable, fields: Collection[str]) -> list[str]:
    """The table's column names: those in `columns` that are among `fields`, else all `fields`."""
    return list(fields) if table.columns is None else [c for c in table.columns if c in fields]


def table_problems(table: ClickHouseTable, owner: str, fields: Collection[str], key: tuple[str, ...]) -> list[str]:
    """The problems in one table built from a landing table with columns `fields` and `PrimaryKey` columns `key`."""
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
    present = set(column_names(table, fields))
    used = [('key', c) for c in key if table.engine == 'rocksdb' and len(key) == 1]
    used += [('order_by', c) for c in table.order_by]
    problems += [f'{where}: {what} {c!r} is not a column of the table' for what, c in used if c not in present]
    return problems


def dictionary_problems(
    dictionary: ClickHouseDictionary, owner: str, tables: Collection[ClickHouseTable], fields: Collection[str]
) -> list[str]:
    """The problems in one dictionary over the entry's `tables`, built from a landing table with columns `fields`."""
    where = f'{owner}.{dictionary.name}'
    source = next((t for t in tables if t.name == dictionary.source), None)
    if source is None:
        return [f'{where}: source {dictionary.source!r} is not a table of the entry']
    known = set(column_names(source, fields))
    return [
        f'{where}: column {c!r} is not a column of {dictionary.source!r}'
        for c in (dictionary.key, *dictionary.attributes)
        if dictionary.source_columns.get(c, c) not in known
    ]
