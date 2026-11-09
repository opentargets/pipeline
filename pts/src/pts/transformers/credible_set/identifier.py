"""Credible set and variant identifiers: what a valid one looks like, and the fields of a variant identifier.

A credible set is identified by the md5 of the columns that define it, and a variant by
`chromosome_position_reference_alternate`. The patterns say what a valid one is, for the QC
stages; the fields are read from a variant identifier, for every rule that locates a variant.
"""

from __future__ import annotations

import polars as pl

from pts.transformers.credible_set.vocabulary import Chromosome

#: A credible set identifier: a 32-character lowercase md5 hex digest.


CREDIBLE_SET_ID_PATTERN = r'^[0-9a-f]{32}$'


VARIANT_ID_PATTERN = rf'^(?:{"|".join(chromosome.value for chromosome in Chromosome)})_[1-9][0-9]*_[ACGT]+_[ACGT]+$'


def variant_chromosome(variant_id: pl.Expr) -> pl.Expr:
    """The chromosome, the first field of `variant_id`.

    Examples:
        >>> pl.select(variant_chromosome(pl.lit('X_100_A_T'))).item()
        'X'
    """
    return _field(variant_id, 0)


def variant_position(variant_id: pl.Expr) -> pl.Expr:
    """The position, the second field of `variant_id`, or null when it is not a number.

    Examples:
        >>> ids = pl.Series(['1_100_A_T', '1_KI270706v1_random_1_A_T'])
        >>> pl.select(variant_position(pl.lit(ids))).to_series().to_list()
        [100, None]
    """
    return _field(variant_id, 1).cast(pl.Int64, strict=False)


def variant_reference_allele(variant_id: pl.Expr) -> pl.Expr:
    """The reference allele, the third field of `variant_id`."""
    return _field(variant_id, 2)


def variant_alternate_allele(variant_id: pl.Expr) -> pl.Expr:
    """The alternate allele, the fourth field of `variant_id`."""
    return _field(variant_id, 3)


def _field(variant_id: pl.Expr, index: int) -> pl.Expr:
    return variant_id.str.split('_').list.get(index, null_on_oob=True)
