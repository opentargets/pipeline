"""Read tables out of an Ensembl core database dump directory.

Ensembl publishes each core database on its FTP site as one tab-separated ``.txt.gz`` per table
plus a single ``.sql.gz`` holding the ``CREATE TABLE`` statements. Despite living under a
directory called ``mysql/`` it is not an archive that a server has to restore: the files are
flat, so polars reads them directly and nothing here starts a database.

The dump being self-describing is the whole reason this module exists. Column names come from
the DDL of the release being read rather than from a list maintained here, so an upstream column
insertion cannot silently shift every field one position to the left -- the failure mode a
hand-maintained positional schema has and cannot detect.
"""

from __future__ import annotations

import gzip
import re
from collections.abc import Mapping
from functools import cached_property

import polars as pl
from otter.storage.synchronous.handle import StorageHandle
from otter.util.errors import OtterError

#: How MySQL's ``SELECT ... INTO OUTFILE`` format spells a NULL: backslash, then N.
NULL_TOKEN = '\\N'  # noqa: S105 not a password, this is the literal null marker in the dump

_SCHEMA_GLOB = '*.sql.gz'
_CREATE_TABLE = re.compile(r'CREATE TABLE `(?P<table>\w+)` \((?P<body>.*?)\n\) ENGINE', re.DOTALL)
#: A column line starts with whitespace then a backtick. Index lines (``PRIMARY KEY``, ``KEY``)
#: put a keyword before their backtick, so they do not match.
_COLUMN = re.compile(r'^\s+`(?P<name>\w+)` ', re.MULTILINE)
_RELEASE = re.compile(r'_core_(?P<release>\d+)_')

#: Placeholder held by an already-unescaped backslash while the other escapes are resolved.
#: Without it, unescaping in sequence turns a literal backslash followed by `t` into a tab.
_SENTINEL = '\x00'


def _unescape(column: str) -> pl.Expr:
    r"""Undo MySQL's outfile escaping on one string column.

    The outfile format writes a literal backslash as ``\\``, a tab as ``\t`` and a newline as
    ``\n``. Resolving those in sequence is wrong -- ``\\t`` (a literal backslash, then the
    letter t) would become a tab -- so an already-unescaped backslash is parked on a sentinel
    until the other forms have been resolved.
    """
    return (
        pl.col(column)
        .str.replace_all('\\\\', _SENTINEL, literal=True)
        .str.replace_all('\\t', '\t', literal=True)
        .str.replace_all('\\n', '\n', literal=True)
        .str.replace_all(_SENTINEL, '\\', literal=True)
    )


class CoreDumpError(OtterError):
    """Raise when a core dump cannot be read as expected."""


class CoreDump:
    """Reads tables out of an Ensembl core database dump directory.

    Args:
        source: Directory holding the ``.txt.gz`` table files and the ``.sql.gz`` schema.
    """

    def __init__(self, source: str) -> None:
        self._source = source.rstrip('/')

    @cached_property
    def _schema_path(self) -> str:
        """Find the single schema file in the dump directory.

        Raises:
            CoreDumpError: If no ``*.sql.gz`` file is present, or more than one is. A stale FTP
                sync or a copy started between two releases could leave a second schema file
                behind, and picking one of them without complaint would silently parse the wrong
                release's columns.
        """
        matches = sorted(StorageHandle(self._source).glob(_SCHEMA_GLOB))
        if not matches:
            raise CoreDumpError(f'no {_SCHEMA_GLOB} schema file in {self._source}')
        if len(matches) > 1:
            raise CoreDumpError(
                f'{len(matches)} {_SCHEMA_GLOB} schema files in {self._source}, expected 1: {matches}'
            )
        return matches[0]

    @cached_property
    def _tables(self) -> dict[str, list[str]]:
        with StorageHandle(self._schema_path).open('rb') as raw, gzip.open(raw, 'rt', encoding='utf8') as ddl:
            text = ddl.read()
        return {m['table']: _COLUMN.findall(m['body']) for m in _CREATE_TABLE.finditer(text)}

    @property
    def release(self) -> str:
        """The Ensembl release this dump is from, taken from the schema filename.

        Raises:
            CoreDumpError: If the dump directory has no schema file, more than one, or the
                schema filename does not carry a release number.
        """
        match = _RELEASE.search(self._schema_path)
        if not match:
            raise CoreDumpError(f'cannot read a release number out of {self._schema_path}')
        return match['release']

    def columns(self, table: str) -> list[str]:
        """The table's columns, in the order the dump writes them.

        Args:
            table: Name of the table to look up.

        Returns:
            Column names in DDL order.

        Raises:
            CoreDumpError: If the dump directory has no schema file, more than one, or `table`
                is not declared in the schema.
        """
        if table not in self._tables:
            raise CoreDumpError(f'{table} is not in {self._schema_path}; it has {sorted(self._tables)}')
        return self._tables[table]

    def scan(self, table: str, columns: Mapping[str, pl.DataType]) -> pl.LazyFrame:
        """Lazily read one table, projected to `columns`.

        The full positional schema is built from the DDL so that polars parses the file by
        position, and only the requested columns are then selected. Requesting a column the
        release does not have raises rather than reading the wrong one.

        Args:
            table: Table name, without the `.txt.gz` suffix.
            columns: Column name to the dtype to read it as. Every `pl.String` column is
                unescaped; other dtypes cannot carry an escape.

        Returns:
            LazyFrame of the requested columns.

        Raises:
            CoreDumpError: If the table or any requested column is not in this release.
        """
        known = self.columns(table)
        missing = [name for name in columns if name not in known]
        if missing:
            raise CoreDumpError(f'{table} in release {self.release} has no column(s) {missing}; it has {known}')

        frame = pl.scan_csv(
            f'{self._source}/{table}.txt.gz',
            schema=pl.Schema({name: columns.get(name, pl.String) for name in known}),
            separator='\t',
            has_header=False,
            null_values=[NULL_TOKEN],
            quote_char=None,
        ).select(columns.keys())

        unescaped = [_unescape(name) for name, dtype in columns.items() if dtype == pl.String]
        return frame.with_columns(unescaped) if unescaped else frame
