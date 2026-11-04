"""The fields of a `chromosome_position_reference_alternate` variant identifier."""

import polars as pl


def _field(variant_id: pl.Expr, index: int) -> pl.Expr:
    return variant_id.str.split('_').list.get(index, null_on_oob=True)


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
