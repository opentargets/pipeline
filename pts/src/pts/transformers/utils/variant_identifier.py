"""Stable identifiers for variants with missing coordinates or long alleles."""

import polars as pl
import polars_hash as plh


def hash_long_variant_ids(frame: pl.LazyFrame, hash_threshold: int = 300) -> pl.LazyFrame:
    """Hash incomplete identifiers, or long IDs retaining their known coordinates."""
    digest = plh.col('variantId').nchash.md5()
    return frame.with_columns(
        pl
        .when(pl.col('chromosome').is_null() | pl.col('position').is_null())
        .then(pl.concat_str(pl.lit('OTVAR_'), digest))
        .when(pl.col('variantId').str.len_chars() >= hash_threshold)
        .then(pl.concat_str(pl.lit('OTVAR'), pl.col('chromosome'), pl.col('position'), digest, separator='_'))
        .otherwise(pl.col('variantId'))
        .alias('variantId')
    )
