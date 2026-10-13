"""The variant search index, and the variant labels the target index reuses."""

from __future__ import annotations

import polars as pl

from pts.transformers.search.helpers import as_list, flatten_cat, search_index


def non_gnomad_xref_ids() -> pl.Expr:
    """The `dbXrefs` ids, excluding gnomad.

    A gnomad xref id (`1-14677-G-A`) is the dash-delimited variantId, 100% redundant with
    `variantId` once OpenSearch normalises delimiters.

    The comparison is NULL-SAFE, matching spark's `eqNullSafe`: an xref whose source is null is
    KEPT. A plain `!=` would evaluate to null there and silently drop it.
    """
    return pl.col('dbXrefs').list.eval(
        pl.element().filter(pl.element().struct.field('source').ne_missing('gnomad')).struct.field('id')
    )


def _chr_prefixed(expr: pl.Expr) -> pl.Expr:
    """Spark's `concat('chr', x)`, which propagates null rather than yielding 'chr'."""
    return pl.concat_str([pl.lit('chr'), expr], ignore_nulls=False)


def variant_labels(variants: pl.LazyFrame) -> pl.LazyFrame:
    """Map each variant to its searchable labels.

    A pure function of the variant row -- it does NOT depend on transcript consequences. That
    is what lets the target index compute these once over 8.17M variants instead of carrying
    them through a 290M-row explode.

    Args:
        variants: frame with `variantId`, `hgvsId`, `dbXrefs`, `rsIds`.

    Returns:
        LazyFrame with `variantId` and `variant_labels`.
    """
    return variants.select(
        'variantId',
        flatten_cat(
            as_list(pl.col('variantId')),
            as_list(_chr_prefixed(pl.col('variantId'))),
            as_list(pl.col('hgvsId')),
            non_gnomad_xref_ids(),
            pl.col('rsIds'),
        ).alias('variant_labels'),
    )


def build_variant_index(variants: pl.LazyFrame) -> pl.LazyFrame:
    """Build the search index for variants.

    Ports `Transformers.Implicits.setIdAndSelectFromVariants`.

    Args:
        variants: frame with `variantId`, `chromosome`, `position`, `rsIds`, `hgvsId`, `dbXrefs`.

    Returns:
        Search index LazyFrame, one row per variant.
    """
    location = pl.concat_str(
        [pl.col('chromosome'), pl.lit('_'), pl.col('position').cast(pl.String), pl.lit('_')],
        ignore_nulls=False,
    )
    frame = variants.with_columns(
        _chr_prefixed(pl.col('variantId')).alias('chrVariantId'),
        location.alias('locationUnderscore'),
    ).with_columns(_chr_prefixed(pl.col('locationUnderscore')).alias('chrLocationUnderscore'))

    identifiers = (
        as_list(pl.col('variantId')),
        as_list(pl.col('hgvsId')),
        non_gnomad_xref_ids(),
        pl.col('rsIds'),
        as_list(pl.col('locationUnderscore')),
        as_list(pl.col('chrVariantId')),
        as_list(pl.col('chrLocationUnderscore')),
    )

    return search_index(
        frame,
        id_col=pl.col('variantId'),
        name_col=pl.col('variantId'),
        entity='variant',
        category_col=as_list(pl.lit('variant')),
        keywords_col=flatten_cat(*identifiers),
        prefixes_col=flatten_cat(*identifiers),
        ngrams_col=flatten_cat(as_list(pl.col('variantId')), non_gnomad_xref_ids()),
        multiplier_col=pl.lit(1.0, dtype=pl.Float64),
    )
