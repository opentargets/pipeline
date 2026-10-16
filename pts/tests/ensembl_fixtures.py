"""Write small Ensembl core dumps for tests."""

from __future__ import annotations

import gzip
from pathlib import Path

NULL = '\\N'
"""How the dump spells NULL."""


def write_gz(path: Path, text: str) -> None:
    """Write `text` to `path` gzipped, as the FTP site ships it."""
    with gzip.open(path, 'wt', encoding='utf8') as handle:
        handle.write(text)


def ddl(tables: dict[str, list[str]]) -> str:
    """`CREATE TABLE` statements for `tables` (name to columns in order), with an index line each."""
    statements = []
    for name, columns in tables.items():
        lines = [f'  `{column}` text,' for column in columns]
        body = '\n'.join([*lines, f'  PRIMARY KEY (`{columns[0]}`)'])
        statements.append(f'CREATE TABLE `{name}` (\n{body}\n) ENGINE=MyISAM DEFAULT CHARSET=latin1;\n')
    return ''.join(statements)


def cells(*values: object) -> list[str]:
    """One dump row: `None` becomes the null token, everything else `str()`."""
    return [NULL if value is None else str(value) for value in values]


def write_dump(
    tmp_path: Path,
    tables: dict[str, list[list[str]]],
    ddl: str,
    *,
    db: str = 'homo_sapiens_core_116_38',
) -> str:
    """Write an Ensembl core dump into `tmp_path`.

    Args:
        tmp_path: Directory to write into.
        tables: Table name to its rows, each row the already-escaped field strings in DDL order.
        ddl: Contents of the schema file.
        db: Database name, which names the schema file `<db>.sql.gz`.

    Returns:
        `tmp_path` as a string, ready for `CoreDump`.
    """
    write_gz(tmp_path / f'{db}.sql.gz', ddl)
    for name, rows in tables.items():
        write_gz(tmp_path / f'{name}.txt.gz', ''.join('\t'.join(row) + '\n' for row in rows))
    return str(tmp_path)
