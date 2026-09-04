"""Shared expression helpers for the search index builders.

Ports `Helpers.flattenCat` and the `SearchIndex` case class from the Scala Search step, by way
of the pyspark job this replaces. Several helpers exist purely to hold a spark/polars
divergence in ONE place -- see the divergence register in the design document.
"""

from __future__ import annotations

import polars as pl

#: The dtype every search array field carries.
LIST_STR = pl.List(pl.String)

#: An empty `List(String)` literal. Used as a default and as the null-fill in `flatten_cat`.
EMPTY_LIST = pl.lit([], dtype=LIST_STR)

#: The four synonym fields on `disease.synonyms`, in the order the pyspark job merged them.
#: Defined here rather than in each consumer so the disease LUT and the disease index cannot
#: drift apart on which flavours they include.
SYNONYM_FLAVOURS = ('hasBroadSynonym', 'hasExactSynonym', 'hasNarrowSynonym', 'hasRelatedSynonym')


def as_list(expr: pl.Expr) -> pl.Expr:
    """Spark's `array(x)` for a scalar column: a one-element list.

    A null scalar becomes `[null]`, exactly as spark's `array(null)` does. `flatten_cat` drops
    the null element later, so a null name contributes nothing either way.
    """
    return pl.concat_list(expr.cast(pl.String))


def list_struct_field(column: str, field: str) -> pl.Expr:
    """Spark's `col.field` on an `array<struct<…>>`, e.g. `synonyms.label`."""
    return pl.col(column).list.eval(pl.element().struct.field(field))


def synonym_field(field: str) -> pl.Expr:
    """Spark's `synonyms.hasBroadSynonym` on a `struct<…: array<string>>`."""
    return pl.col('synonyms').struct.field(field)


def flatten_cat(*exprs: pl.Expr) -> pl.Expr:
    """Build a deduplicated, null-filtered, comma-stripped list from list expressions.

    Ports `Helpers.flattenCat`. The spark original is:

        filter(array_distinct(transform(flatten(filter(array(cols), x -> isnotnull(x))),
               s -> replace(trim(s), ',', ''))), t -> isnotnull(t))

    Three divergences are handled here rather than at the 24 call sites:

    * **A null input list must be dropped, not propagated.** `pl.concat_list` returns null if
      ANY input is null, whereas spark's inner `filter(…, isnotnull)` removes the null arrays
      and keeps the rest. Without the per-input `fill_null` a single null synonym field would
      empty the whole keywords array for that row.
    * **`trim` strips 0x20 only.** polars' bare `strip_chars()` strips all unicode whitespace,
      which would mangle any label carrying a non-breaking space.
    * **`array_distinct` keeps the first occurrence**, so dedup must maintain order.

    Args:
        *exprs: expressions each yielding `List(String)`. Wrap scalars with `as_list`.

    Returns:
        An expression yielding a deduplicated `List(String)` with nulls removed.
    """
    filled = [expr.cast(LIST_STR).fill_null(EMPTY_LIST) for expr in exprs]
    return (
        pl.concat_list(filled)
        .list.eval(pl.element().str.strip_chars(' ').str.replace_all(',', '', literal=True))
        .list.unique(maintain_order=True)
        .list.drop_nulls()
    )


def tier(values: str, rank: str, limit: int) -> pl.Expr:
    """Aggregate the label lists of the rows within a rank cutoff.

    Ports `array_distinct(flatten(collect_list(when(rank <= limit, values))))`, for use inside
    a `group_by(...).agg(...)`.

    `keep_nulls=False` on the explode is `collect_list`'s null-dropping: a row whose label list
    is null contributes nothing rather than nulling the aggregate. The trailing `drop_nulls`
    removes any null string elements that survive inside an otherwise non-null list, mirroring
    spark's final `filter(t -> isnotnull(t))`. The result is `[]`, never null, when no row falls
    inside the limit -- the equivalence bar treats those as different values.

    Args:
        values: name of the `List(String)` column to aggregate.
        rank: name of the rank column to filter on.
        limit: inclusive rank cutoff.

    Returns:
        An aggregation expression yielding a deduplicated `List(String)`.
    """
    return (
        pl.col(values)
        .filter(pl.col(rank) <= limit)
        .list.explode(keep_nulls=False, empty_as_null=False)
        .unique(maintain_order=True)
        .drop_nulls()
    )


def search_index(
    lf: pl.LazyFrame,
    *,
    id_col: pl.Expr,
    name_col: pl.Expr,
    entity: str,
    category_col: pl.Expr,
    keywords_col: pl.Expr,
    prefixes_col: pl.Expr,
    ngrams_col: pl.Expr,
    description_col: pl.Expr | None = None,
    terms_col: pl.Expr | None = None,
    terms25_col: pl.Expr | None = None,
    terms5_col: pl.Expr | None = None,
    multiplier_col: pl.Expr | None = None,
) -> pl.LazyFrame:
    """Project a frame onto the release's search-index schema.

    Ports the `SearchIndex` case class. Column ORDER is part of the output contract, so this is
    the only place any of the five builders is allowed to name these columns.

    Args:
        lf: source frame.
        id_col: entity id.
        name_col: display name.
        entity: entity type string, e.g. `'disease'`.
        category_col: category list.
        keywords_col: keywords list.
        prefixes_col: prefixes list.
        ngrams_col: ngrams list.
        description_col: description; defaults to a typed null.
        terms_col: top-50 terms; defaults to an empty list.
        terms25_col: top-25 terms; defaults to an empty list.
        terms5_col: top-5 terms; defaults to an empty list.
        multiplier_col: ranking multiplier; defaults to `0.01`.

    Returns:
        LazyFrame carrying exactly the twelve release columns, in order.
    """
    return lf.select(
        id_col.alias('id'),
        name_col.alias('name'),
        (pl.lit(None, dtype=pl.String) if description_col is None else description_col).alias('description'),
        pl.lit(entity, dtype=pl.String).alias('entity'),
        category_col.alias('category'),
        keywords_col.alias('keywords'),
        prefixes_col.alias('prefixes'),
        ngrams_col.alias('ngrams'),
        (EMPTY_LIST if terms_col is None else terms_col).alias('terms'),
        (EMPTY_LIST if terms25_col is None else terms25_col).alias('terms25'),
        (EMPTY_LIST if terms5_col is None else terms5_col).alias('terms5'),
        (pl.lit(0.01, dtype=pl.Float64) if multiplier_col is None else multiplier_col).alias('multiplier'),
    )
