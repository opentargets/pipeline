"""Polars equivalent of PySpark Evidence's variant identifier hashing."""

import polars as pl
import polars_hash as plh

from pts.transformers.utils.spark_strings import regexp_extract

VARIANT_HASH_LENGTH = 300


def hash_long_variant_ids(variant_id: pl.Expr, threshold: int = VARIANT_HASH_LENGTH) -> pl.Expr:
    """Hash IDs longer than threshold, preserving Spark Evidence's behavior.

    Extract chromosome and position from canonical IDs. Nonmatching non-null
    IDs use empty coordinate strings, matching Spark's regexp_extract; long
    nonmatching IDs therefore become OTVAR___<md5>. Null IDs remain null.

    Examples:
        IDs at the threshold stay unchanged; longer IDs retain their coordinates.

        >>> import hashlib
        >>> boundary_id = '1_10_A_' + 'T' * 293
        >>> long_id = boundary_id + 'T'
        >>> frame = pl.DataFrame({'variantId': ['1_10_A_T', boundary_id, long_id, None]})
        >>> result = frame.select(hash_long_variant_ids(pl.col('variantId'))).to_series().to_list()
        >>> result == ['1_10_A_T', boundary_id, 'OTVAR_1_10_' + hashlib.md5(long_id.encode()).hexdigest(), None]
        True
    """
    pattern = r'^([0-9XYMT]{1,2})_([0-9]+)_([ACGTN]+)_([ACGTN]+)$'

    chrom = regexp_extract(variant_id, pattern, 1)
    position = regexp_extract(variant_id, pattern, 2)
    digest = plh.col('variantId').nchash.md5()
    return (
        pl
        .when(chrom.is_null() | position.is_null())
        .then(pl.lit('OTVAR_') + digest)
        .when(variant_id.str.len_chars() > threshold)
        .then(pl.concat_str(pl.lit('OTVAR'), chrom, position, digest, separator='_'))
        .otherwise(variant_id)
    )
