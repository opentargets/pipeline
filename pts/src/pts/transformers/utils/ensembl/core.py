"""Read tables from an Ensembl core database dump directory."""

from __future__ import annotations

import gzip
import re
from collections.abc import Mapping
from functools import cached_property
from pathlib import Path

import polars as pl
from otter.storage.synchronous.handle import StorageHandle
from otter.util.errors import OtterError

NULL_TOKEN = '\\N'  # noqa: S105 not a password, the literal null marker in the dump

_SCHEMA_GLOB = '*.sql.gz'
_CREATE_TABLE = re.compile(r'CREATE TABLE `(?P<table>\w+)` \((?P<body>.*?)\n\) ENGINE', re.DOTALL)
_COLUMN = re.compile(r'^\s+`(?P<name>\w+)` ', re.MULTILINE)
_RELEASE = re.compile(r'_core_(?:\d+_)?(?P<release>\d+)_\d+\.sql\.gz$')

_ESCAPES = {
    '\\t': '\t',
    '\\n': '\n',
    '\\r': '\r',
    '\\0': '\x00',
    '\\b': '\b',
    '\\Z': '\x1a',
}


def _unescape(column: str) -> pl.Expr:
    r"""Resolve MySQL outfile escapes in a string column; ``\\`` is a literal backslash."""
    piece = pl.element()
    for escaped, char in _ESCAPES.items():
        piece = piece.str.replace_all(escaped, char, literal=True)
    return pl.col(column).str.split('\\\\').list.eval(piece).list.join('\\').alias(column)


class CoreDumpError(OtterError):
    """Raise when a core dump cannot be read as expected."""


class CoreDump:
    """Reads tables out of an Ensembl core database dump directory.

    Args:
        source: Directory holding the ``<table>.txt.gz`` files and exactly one ``*.sql.gz`` schema.
    """

    def __init__(self, source: str) -> None:
        self._source = source.rstrip('/')

    @cached_property
    def _schema_path(self) -> str:
        matches = sorted(StorageHandle(self._source).glob(_SCHEMA_GLOB))
        if not matches:
            raise CoreDumpError(f'no {_SCHEMA_GLOB} schema file in {self._source}')
        if len(matches) > 1:
            raise CoreDumpError(f'{len(matches)} {_SCHEMA_GLOB} schema files in {self._source}, expected 1: {matches}')
        return matches[0]

    @cached_property
    def _tables(self) -> dict[str, list[str]]:
        with StorageHandle(self._schema_path).open('rb') as raw, gzip.open(raw, 'rt', encoding='utf8') as ddl:
            text = ddl.read()
        return {m['table']: _COLUMN.findall(m['body']) for m in _CREATE_TABLE.finditer(text)}

    @property
    def release(self) -> str:
        """The Ensembl release, read from the schema filename (``homo_sapiens_core_116_38.sql.gz`` -> ``116``).

        Raises:
            CoreDumpError: If the schema filename carries no release number.
        """
        match = _RELEASE.search(Path(self._schema_path).name)
        if not match:
            raise CoreDumpError(f'cannot read a release number out of {self._schema_path}')
        return match['release']

    def columns(self, table: str) -> list[str]:
        """The table's columns, in the order the dump writes them.

        Raises:
            CoreDumpError: If `table` is not declared in the schema.
        """
        if table not in self._tables:
            raise CoreDumpError(f'{table} is not in release {self.release}; it has {sorted(self._tables)}')
        return self._tables[table]

    def scan(self, table: str, columns: Mapping[str, pl.DataType]) -> pl.LazyFrame:
        """Lazily read one table, projected to `columns`.

        Args:
            table: Table name, without the ``.txt.gz`` suffix.
            columns: Column name to the dtype to read it as. `pl.String` columns are unescaped.

        Returns:
            LazyFrame of the requested columns, in the requested order.

        Raises:
            CoreDumpError: If the table or any requested column is not in this release.
        """
        known = self.columns(table)
        missing = [name for name in columns if name not in known]
        if missing:
            raise CoreDumpError(f'{table} in release {self.release} has no column(s) {missing}; it has {known}')

        frame = pl.scan_csv(
            f'{self._source}/{table}.txt.gz',
            schema=pl.Schema({name: columns.get(name, pl.String()) for name in known}),
            separator='\t',
            has_header=False,
            null_values=[NULL_TOKEN],
            quote_char=None,
            empty_string_is_null=False,
        ).select(list(columns))

        unescaped = [_unescape(name) for name, dtype in columns.items() if dtype == pl.String]
        return frame.with_columns(unescaped) if unescaped else frame
