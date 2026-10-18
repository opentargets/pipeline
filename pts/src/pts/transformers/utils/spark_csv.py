r"""A CSV reader that parses the way Spark's does.

Spark reads one record per line (a quoted newline splits the record), takes `"` as the quote and `\`
as the escape, and leaves a malformed quoted field as text up to the next separator. Polars' reader
differs on each, so a source read by spark in the old steps is read with this.
"""

from __future__ import annotations

import itertools
import math
import re

import polars as pl
from otter.storage.synchronous.handle import StorageHandle

from pts.transformers.utils.spark_casts import SPARK_CAST_TRIM

QUOTE = '"'
ESCAPE = '\\'

_LINE_END = re.compile(r'\r\n|\r|\n')
#: java's `String.trim` characters: everything up to and including the space.
_BLANK = re.compile(r'[\x00- ]*\Z')

_ASCII_INT = r'^[+-]?[0-9]+$'
#: `Integer.parseInt` takes any unicode decimal digit
_JAVA_INT = r'^[+-]?\d+$'
_INT32_MIN, _INT32_MAX = -(2**31), 2**31 - 1
#: java's decimal syntax, as `spark_casts._float_text` reads it
_JAVA_DECIMAL = r'^([+-]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?)[fFdD]?$'
_JAVA_HEX = r'^([+-]?0[xX](?:[0-9a-fA-F]+\.?[0-9a-fA-F]*|\.[0-9a-fA-F]+)[pP][+-]?[0-9]+)[fFdD]?$'


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
    columns: list[str] | None = None,
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
        columns: the names to keep, after header renaming or `new_columns`; the result has them in this order.

    Returns:
        DataFrame of strings.

    Raises:
        ValueError: if `columns` is empty, repeats a name, or names something that is not a column.
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
    rows = (_fields(line, separator) for line in lines)
    if names is None:
        first_row = next(rows, None)
        names = [f'_c{i}' for i in range(0 if first_row is None else len(first_row))]
        rows = itertools.chain([] if first_row is None else [first_row], rows)
    width = len(names)

    if columns is None:
        keep = list(range(width))
    else:
        if not columns:
            raise ValueError('columns is empty')
        repeated = sorted({c for c in columns if columns.count(c) > 1})
        if repeated:
            raise ValueError(f'columns repeated: {repeated}')
        unknown = [c for c in columns if c not in names]
        if unknown:
            raise ValueError(f'columns not in the file: {unknown}')
        keep = [names.index(c) for c in columns]
        names = columns

    def clean(value: str | None) -> str | None:
        return None if value is None or value in nulls else value

    padded = [[clean(row[i]) if i < len(row) else None for i in keep] for row in rows]
    return pl.DataFrame(padded, schema=dict.fromkeys(names, pl.String), orient='row')


def _java_int(text: str) -> int | None:
    """Unicode decimal digits as an int, null outside the Int32 range."""
    value = int(text)
    return value if _INT32_MIN <= value <= _INT32_MAX else None


def _unicode_ints(texts: pl.Series) -> pl.Series:
    """The texts of non-ASCII decimal digits as Int32, null elsewhere."""
    unicode = (texts.str.contains(_JAVA_INT) & ~texts.str.contains(_ASCII_INT)).fill_null(False)
    out = pl.Series(texts.name, [None] * len(texts), dtype=pl.Int32)
    rows = unicode.arg_true()
    if rows.is_empty():
        return out
    return out.scatter(rows, pl.Series([_java_int(t) for t in texts.gather(rows).to_list()], dtype=pl.Int32))


def spark_csv_int(field: pl.Expr) -> pl.Expr:
    """An IntegerType field as Spark's CSV reader parses it (java's `Integer.parseInt`), null if it fails, Int32."""
    return (
        pl
        .when(field.str.contains(_ASCII_INT))
        .then(field.str.strip_prefix('+').cast(pl.Int64, strict=False).cast(pl.Int32, strict=False))
        .when(field.str.contains(_JAVA_INT))
        .then(field.map_batches(_unicode_ints, return_dtype=pl.Int32, is_elementwise=True))
    )


def _from_hex(texts: pl.Series) -> pl.Series:
    """Hexadecimal float texts as doubles, null for null."""
    values = texts.to_list()
    return pl.Series(texts.name, [None if t is None else float.fromhex(t) for t in values], dtype=pl.Float64)


def spark_csv_double(field: pl.Expr) -> pl.Expr:
    """A DoubleType field as Spark's CSV reader parses it, null if it fails, Float64.

    `NaN`, `Inf` and `-Inf` are Spark's own tokens; anything else goes through java's
    `Double.parseDouble`: trimmed, an optional `f`/`d` suffix, signed `NaN` and `Infinity`, and
    hexadecimal floats.
    """
    trimmed = field.str.strip_chars(SPARK_CAST_TRIM)
    return (
        pl
        .when(field == 'NaN')
        .then(pl.lit(math.nan))
        .when(field == 'Inf')
        .then(pl.lit(math.inf))
        .when(field == '-Inf')
        .then(pl.lit(-math.inf))
        .when(trimmed.str.contains(r'^[+-]?NaN$'))
        .then(pl.lit(math.nan))
        .when(trimmed.str.contains(r'^\+?Infinity$'))
        .then(pl.lit(math.inf))
        .when(trimmed == '-Infinity')
        .then(pl.lit(-math.inf))
        .when(trimmed.str.contains(_JAVA_HEX))
        .then(trimmed.str.extract(_JAVA_HEX, 1).map_batches(_from_hex, return_dtype=pl.Float64, is_elementwise=True))
        .otherwise(trimmed.str.extract(_JAVA_DECIMAL, 1).cast(pl.Float64, strict=False))
    )
