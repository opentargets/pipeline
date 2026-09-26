"""Evidence identifier assignment and duplicate detection.

Polars port of `assign_evidence_identifier`/`validate_uniqueness` in
`pts.pyspark.evidence_utils.evidence.Evidence`. Polars has no built-in SHA hash, so both hashes
are computed row-wise via Python's `hashlib` through `map_elements` -- fine at GWAS/ENCORE evidence
volume, but worth revisiting if a much larger evidence source reuses this.
"""

import hashlib
import json

import polars as pl

from pts.transformers.evidence.utils import flags
from pts.transformers.utils.quality_flags import update_quality_flag


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode('utf-8')).hexdigest()


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _stringify_field(df: pl.DataFrame, field: str) -> pl.Expr:
    """Stringify one column for identifier hashing.

    A plain `.cast(pl.String)` works for scalar columns but Polars refuses it outright for
    List/Struct columns (unlike Spark, which happily stringifies nested types) -- e.g. ENCORE's
    `diseaseCellLines` is a `list[struct]` and is one of its `unique_fields`. Those go through
    `json.dumps` instead for a stable, order-independent-within-a-struct representation; this
    doesn't need to byte-match Spark's own struct-to-string formatting, since `id` here is only a
    within-release deduplication key, never a stable cross-release identifier (see
    `gwas_evidence.py`'s docstring).
    """
    dtype = df.schema[field]
    if isinstance(dtype, pl.List | pl.Struct):
        return (
            pl.col(field)
            .map_elements(lambda value: json.dumps(value, sort_keys=True, default=str), return_dtype=pl.String)
            .fill_null('null')
        )
    return pl.col(field).cast(pl.String).fill_null('null')


def assign_evidence_identifier(df: pl.DataFrame, unique_fields: list[str]) -> pl.DataFrame:
    """Add a stable `id` column derived from `unique_fields`.

    Args:
        df: evidence dataframe.
        unique_fields: column names that define evidence uniqueness. Columns absent from `df`
            are silently skipped, matching the PySpark behaviour.

    Returns:
        `df` with a new `id` column: `sha1` of the `unique_fields` values, concatenated in order,
        each coalesced to the literal string `'null'`.
    """
    present_fields = [field for field in unique_fields if field in df.columns]
    concatenated = pl.concat_str(
        [_stringify_field(df, field) for field in present_fields],
        separator='',
    )
    return df.with_columns(concatenated.map_elements(_sha1, return_dtype=pl.String).alias('id'))


def _content_string(dtype: pl.DataType, column: str) -> pl.Expr:
    """Stringify one column for content hashing.

    Unlike Spark, Polars refuses to `.cast(String)` a `List` or `Struct` column directly
    (`strict_cast` fails outright rather than stringifying). A list of scalars -- e.g.
    `literature` -- is joined element-wise; a column containing a struct anywhere (a bare struct,
    or -- like ENCORE's `biomarkerList`/`diseaseCellLines`/`validationReadouts` -- a list of
    structs) goes through `json.dumps` instead, same as `_stringify_field` above.
    """
    if isinstance(dtype, pl.Struct) or (isinstance(dtype, pl.List) and isinstance(dtype.inner, pl.Struct)):
        return (
            pl.col(column)
            .map_elements(lambda value: json.dumps(value, sort_keys=True, default=str), return_dtype=pl.String)
            .fill_null('null')
        )
    if isinstance(dtype, pl.List):
        return pl.col(column).fill_null([]).cast(pl.List(pl.String)).list.join(',')
    return pl.col(column).cast(pl.String).fill_null('null')


def validate_uniqueness(df: pl.DataFrame) -> pl.DataFrame:
    """Flag all but one row of each duplicate `id` group as `flags.DUPLICATED`.

    The surviving row is picked by a stable content hash over every column, so the same row wins
    regardless of input row order.

    Args:
        df: evidence dataframe with an `id` column.

    Returns:
        `df` with `qualityControls` flagged with `flags.DUPLICATED` on every row of an `id` group
        after the first, ranked by content hash.
    """
    content_columns = [column for column in df.columns if column != 'qualityControls']
    content_hash = pl.concat_str(
        [_content_string(df.schema[column], column) for column in content_columns],
        separator='\x01',
    ).map_elements(_sha256, return_dtype=pl.String)

    ranked = df.with_columns(content_hash.alias('_content_hash')).with_columns(
        pl.col('_content_hash').rank(method='ordinal').over('id').alias('_rank')
    )
    ranked = update_quality_flag(ranked, pl.col('_rank') != 1, flags.DUPLICATED)
    return ranked.drop('_content_hash', '_rank')
