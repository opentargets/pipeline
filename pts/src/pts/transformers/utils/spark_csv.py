r"""A CSV reader that parses the way Spark's does.

Spark reads one record per line (a quoted newline splits the record), takes `"` as the quote and `\`
as the escape, and leaves a malformed quoted field as text up to the next separator. Polars' reader
differs on each, so a source read by spark in the old steps is read with this.
"""

from __future__ import annotations

import re

import polars as pl
from otter.storage.synchronous.handle import StorageHandle

QUOTE = '"'
ESCAPE = '\\'

_LINE_END = re.compile(r'\r\n|\r|\n')
#: java's `String.trim` characters: everything up to and including the space.
_BLANK = re.compile(r'[\x00- ]*\Z')


def _quoted(line: str, i: int, separator: str) -> tuple[str, int]:
    """Parse a quoted field whose opening quote precedes `i`; returns the value and the index of its end."""
    n = len(line)
    out: list[str] = []
    closing = False  # the last character kept was a quote that may be the closing one
    while i < n:
        c = line[i]
        if closing:
            blanks_end = i
            while blanks_end < n and line[blanks_end] <= ' ' and line[blanks_end] != separator:
                blanks_end += 1
            if blanks_end == n or line[blanks_end] == separator:
                out.pop()
                return ''.join(out), blanks_end
            if c != QUOTE:
                # unescaped quote or text after the closing quote: raw text from the opening quote
                end = line.find(separator, i)
                end = n if end < 0 else end
                return QUOTE + ''.join(out) + line[i:end], end
        if c == ESCAPE:
            if i + 1 < n and line[i + 1] in (QUOTE, ESCAPE):
                out.append(line[i + 1])
                i += 2
            elif i + 1 < n:
                out.append(c)
                i += 1
            else:
                i += 1
            closing = False
            continue
        out.append(c)
        closing = c == QUOTE
        i += 1
    if closing:
        out.pop()
    return ''.join(out) if out or closing else QUOTE, n


def _fields(line: str, separator: str) -> list[str | None]:
    """Split a line into fields; an empty unquoted field is None."""
    if QUOTE not in line:
        return [field or None for field in line.split(separator)]
    n = len(line)
    fields: list[str | None] = []
    i = 0
    while True:
        if i < n and line[i] == QUOTE:
            value, i = _quoted(line, i + 1, separator)
            fields.append(value)
        else:
            end = line.find(separator, i)
            end = n if end < 0 else end
            fields.append(line[i:end] or None)
            i = end
        if i >= n:
            return fields
        i += len(separator)


def _safe_names(header: list[str | None], null_values: list[str]) -> list[str]:
    """Spark's header names: blanks become `_c<i>`, and names repeated ignoring case get their index appended."""
    lowered = [h.lower() for h in header if h]
    repeated = {h for h in lowered if lowered.count(h) > 1}
    return [
        f'_c{i}' if not h or h in null_values else f'{h}{i}' if h.lower() in repeated else h
        for i, h in enumerate(header)
    ]


def read_spark_csv(
    path: str,
    *,
    separator: str = ',',
    has_header: bool = True,
    new_columns: list[str] | None = None,
    null_values: list[str] | None = None,
) -> pl.DataFrame:
    r"""Read a CSV file as Spark's reader does with its defaults; every column is a string.

    Blank lines are skipped. A field is null when it is empty and unquoted, or equals one of
    `null_values` (default `['']`, so a quoted `""` is null too). Rows are cut or padded with null to
    the width of `new_columns`, or of the header. A quoted field closes at its quote followed by a
    separator or the line end, with or without whitespace between; `\"` and `\\` inside it are
    escapes, and any other backslash is kept. Any other text after the closing quote makes the field
    the raw text from its opening quote to the next separator.

    Args:
        path: the file to read.
        separator: the field separator.
        has_header: whether the first non-blank line is a header, which is dropped when `new_columns` is given.
        new_columns: names for the columns by position.
        null_values: strings that read as null.

    Returns:
        DataFrame of strings.
    """
    nulls = [''] if null_values is None else null_values
    with StorageHandle(path).open('rb') as f:
        text = f.read().decode('utf-8', errors='replace').removeprefix('\ufeff')
    lines = (line for line in _LINE_END.split(text) if not _BLANK.match(line))

    names = new_columns
    if has_header:
        first = next(lines, None)
        header = [] if first is None else _fields(first, separator)
        if names is None:
            names = _safe_names(header, nulls)
    rows = [_fields(line, separator) for line in lines]
    if names is None:
        names = [f'_c{i}' for i in range(len(rows[0]) if rows else 0)]
    width = len(names)

    def clean(value: str | None) -> str | None:
        return None if value is None or value in nulls else value

    padded = [[clean(v) for v in row[:width]] + [None] * (width - len(row)) for row in rows]
    return pl.DataFrame(padded, schema=dict.fromkeys(names, pl.String), orient='row')
