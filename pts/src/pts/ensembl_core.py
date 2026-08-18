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
from functools import cached_property

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
        matches = sorted(StorageHandle(self._source).glob(_SCHEMA_GLOB))
        if not matches:
            raise CoreDumpError(f'no {_SCHEMA_GLOB} schema file in {self._source}')
        return matches[0]

    @cached_property
    def _tables(self) -> dict[str, list[str]]:
        with StorageHandle(self._schema_path).open('rb') as raw, gzip.open(raw, 'rt', encoding='utf8') as ddl:
            text = ddl.read()
        return {m['table']: _COLUMN.findall(m['body']) for m in _CREATE_TABLE.finditer(text)}

    @property
    def release(self) -> str:
        """The Ensembl release this dump is from, taken from the schema filename."""
        match = _RELEASE.search(self._schema_path)
        if not match:
            raise CoreDumpError(f'cannot read a release number out of {self._schema_path}')
        return match['release']

    def columns(self, table: str) -> list[str]:
        """The table's columns, in the order the dump writes them."""
        if table not in self._tables:
            raise CoreDumpError(f'{table} is not in {self._schema_path}; it has {sorted(self._tables)}')
        return self._tables[table]
