"""Polars equivalent of PySpark Evidence's variant identifier hashing."""

import polars as pl
import polars_hash as plh

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

    def _extract(group: int) -> pl.Expr:
        # Spark returns '' on a non-match, but null when the input itself is null.
        return pl.when(variant_id.is_null()).then(None).otherwise(variant_id.str.extract(pattern, group).fill_null(''))

    chromosome = _extract(1)
    position = _extract(2)
    digest = plh.NonCryptographicHashingNameSpace(variant_id).md5()
    return (
        pl
        .when(chromosome.is_null() | position.is_null())
        .then(pl.lit('OTVAR_') + digest)
        .when(variant_id.str.len_chars() > threshold)
        .then(pl.concat_str(pl.lit('OTVAR'), chromosome, position, digest, separator='_'))
        .otherwise(variant_id)
    )
