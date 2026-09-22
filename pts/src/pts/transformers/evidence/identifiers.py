"""Evidence identifier assignment and duplicate detection.

Polars port of `assign_evidence_identifier`/`validate_uniqueness` in
`pts.pyspark.evidence_utils.evidence.Evidence`. Polars has no built-in SHA hash, so both hashes
are computed row-wise via Python's `hashlib` through `map_elements` -- fine at GWAS evidence
volume, but worth revisiting if a much larger evidence source reuses this.
"""

import hashlib

import polars as pl

from pts.transformers.evidence import flags
from pts.transformers.utils.quality_flags import update_quality_flag


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode('utf-8')).hexdigest()


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


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
        [pl.col(field).cast(pl.String).fill_null('null') for field in present_fields],
        separator='',
    )
    return df.with_columns(concatenated.map_elements(_sha1, return_dtype=pl.String).alias('id'))


def _content_string(dtype: pl.DataType, column: str) -> pl.Expr:
    """Stringify one column for content hashing.

    Unlike Spark, Polars refuses to `.cast(String)` a `List` column directly (`strict_cast`
    fails outright rather than stringifying), so list columns -- e.g. `literature` -- are joined
    element-wise instead.
    """
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
