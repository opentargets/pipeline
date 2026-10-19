"""Polars and text helpers for the clinical report providers."""

import re
from collections.abc import Callable

import polars as pl


def union_dfs(dfs: list[pl.DataFrame]) -> pl.DataFrame:
    """Concatenates a list of Polars DataFrames, filling missing column values with null.

    Args:
        dfs (List[pl.DataFrame]): A list of Polars DataFrames to concatenate.

    Returns:
        pl.DataFrame: A single concatenated DataFrame.
    """
    return pl.concat(dfs, how='diagonal')


def filter_df(df: pl.DataFrame, expr: str | pl.Expr) -> pl.DataFrame:
    """Filters a Polars DataFrame based on a condition.

    Args:
        df (pl.DataFrame): The DataFrame to filter.
        expr (str | pl.Expr): The condition to filter by. Can be a string expression

    Returns:
        pl.DataFrame: The filtered DataFrame.
    """
    if isinstance(expr, pl.Expr):
        return df.filter(expr)

    ctx = pl.SQLContext()
    ctx.register('df', df)

    return ctx.execute(f'SELECT * FROM df WHERE {expr}').collect()  # noqa: S608


def extract_per_distinct_text(
    df: pl.DataFrame,
    input_col: str,
    output_col: str,
    extract: Callable[[pl.Series], pl.Series],
) -> pl.DataFrame:
    """Append ``output_col`` with ``extract`` applied once per distinct non-null text of ``input_col``.

    Args:
        df: frame holding the texts.
        input_col: String column with the texts.
        output_col: name of the appended ``List(String)`` column; an empty list for null texts.
        extract: maps a String series of texts to a ``List(String)`` series of the same length.

    Returns:
        ``df`` with ``output_col`` appended.
    """
    texts = df.get_column(input_col).drop_nulls().unique().sort()
    extracted = extract(texts)
    if extracted.len() != texts.len():
        raise ValueError(f'extraction returned {extracted.len()} rows for {texts.len()} texts')
    lookup = pl.DataFrame({input_col: texts, output_col: extracted.cast(pl.List(pl.String))})
    return df.join(lookup, on=input_col, how='left', maintain_order='left').with_columns(
        pl.col(output_col).fill_null(pl.lit([], dtype=pl.List(pl.String)))
    )


def sanitise_text(value: str | None) -> str | None:
    """Sanitise a string by removing null bytes and escaped characters."""
    if value is None:
        return None
    value = re.sub(r'\x00([0-9A-Fa-f]{2})', lambda m: chr(int(m.group(1), 16)), value)
    value = value.replace('\x00', '')
    return re.sub(r'\\([\[\]><&~_^*])', r'\1', value)


def sanitise_nested(obj):
    """Recursively sanitise a nested object (dict/list/str)."""
    if isinstance(obj, str):
        return sanitise_text(obj)
    if isinstance(obj, dict):
        return {k: sanitise_nested(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitise_nested(v) for v in obj]
    return obj
