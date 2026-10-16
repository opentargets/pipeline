"""Polars port of `OnToma.map_entities`: resolve labels and ids against an OnToma lookup table.

The lookup tables themselves are still built by OnToma on spark (`pyspark ontoma_lut_generation`);
only the query side is ported. It must normalise exactly as the tables were, so it is tied to the
spark-nlp version that built them -- see `pts.transformers.utils.ontoma.normalise`.

Behaviour kept from OnToma on purpose, because it is observable in the output:

* **Identical input rows collapse into one.** OnToma aggregates by every input column, so
  duplicates do not survive the mapping. This port deduplicates the same way.
* **Each label is tried in two spellings**, translated with and without accented letters, and
  the identifiers found for either are merged.
* **No identifier means null, not an empty list.**

Differences, neither of which can change which identifiers are found:

* The identifier lists are sorted. OnToma's come out of `collect_set` in no particular order.
* Rows keep their input order, which spark's aggregation does not.

Diffed against `OnToma.map_entities` on spark with lookup tables built by OnToma from the disease
index: 162,085 query rows covering names, synonyms, cross references and perturbed copies of
them, mapped as labels and as ids, with a type column of mixed and null types and with both
`include_*` flags, matched exactly in every configuration.
"""

from __future__ import annotations

from typing import Literal

import polars as pl

from pts.transformers.utils.dataset import scan_dataset
from pts.transformers.utils.ontoma.normalise import normalise
from pts.transformers.utils.ontoma.query import (
    alternative_translations,
    clean_disease_label,
    determine_track,
    format_identifier,
    spark_trim,
)

EntityKind = Literal['label', 'id']

#: Column of the normalised label, in the lookup table and in the output when it is asked for.
NORMALISED_COLUMN = 'entityLabelNormalised'

_ROW = '__ontoma_row'
_ENTITY = '__ontoma_entity'
_TYPE = '__ontoma_type'
_LABEL = '__ontoma_label'
_TRACK = '__ontoma_track'


def read_entity_lut(path: str) -> pl.LazyFrame:
    """Read a lookup table OnToma wrote (`ReadyEntityLUT`), as written by its `cache_dir`."""
    return scan_dataset(path).select('entityKind', 'entityType', NORMALISED_COLUMN, 'entityIds')


def _normalise_batch(batch: pl.Series) -> pl.Series:
    labels = batch.struct.field(_LABEL).to_list()
    tracks = batch.struct.field(_TRACK).to_list()
    return pl.Series([normalise(label, track) for label, track in zip(labels, tracks, strict=True)], dtype=pl.String)


def _query_labels(query: pl.LazyFrame) -> pl.LazyFrame:
    """`OnToma._extract_query_entity_labels`."""
    entity = pl.col(_ENTITY)
    entity_type = pl.col(_TYPE)
    return (
        query
        .with_columns(
            alternative_translations(spark_trim(entity)).alias(_LABEL),
            # decided on the raw label, before trimming, translating or cleaning
            pl.when(entity_type == 'CD').then(pl.lit('symbol')).otherwise(determine_track(entity)).alias(_TRACK),
        )
        .explode(_LABEL)
        .with_columns(
            pl
            .when(entity_type == 'DS')
            .then(clean_disease_label(pl.col(_LABEL)))
            .otherwise(pl.col(_LABEL))
            .alias(_LABEL),
        )
    )


def _query_ids(query: pl.LazyFrame) -> pl.LazyFrame:
    """`OnToma._extract_query_entity_ids`."""
    return query.with_columns(
        format_identifier(spark_trim(pl.col(_ENTITY)).str.to_uppercase()).alias(_LABEL),
        pl.lit('symbol').alias(_TRACK),
    )


def _normalise_query(query: pl.LazyFrame) -> pl.LazyFrame:
    """`OnToma._normalise_entities`: normalise each distinct label once and join the result back.

    The join is an equi-join as in OnToma, so a null label stays unnormalised.
    """
    distinct = (
        query
        .select(_LABEL, _TRACK)
        .unique()
        .with_columns(
            pl.struct(_LABEL, _TRACK).map_batches(_normalise_batch, return_dtype=pl.String).alias(NORMALISED_COLUMN),
        )
    )
    return query.join(distinct, on=[_LABEL, _TRACK], how='left')


def map_entities(
    lf: pl.LazyFrame,
    lut: pl.LazyFrame,
    result_col_name: str,
    entity_col_name: str,
    entity_kind: EntityKind,
    type_col_name: str | None = None,
    type_col: pl.Expr | None = None,
    *,
    include_normalised_entities: bool = False,
    include_entity_source: bool = False,
) -> pl.LazyFrame:
    """Map the labels or ids in a column to entity identifiers, as `OnToma.map_entities` does.

    Args:
        lf: Frame holding the entities to map.
        lut: Lookup table, as `read_entity_lut` returns it.
        result_col_name: Name of the column added for the identifiers found.
        entity_col_name: Column holding the label or id to map.
        entity_kind: `label` or `id`; only lookup table rows of that kind are matched.
        type_col_name: Column holding each entity's type (`DS`, `CD`, `GP`, ...).
        type_col: Expression giving the type instead, e.g. `pl.lit('DS')`; it is not kept.
        include_normalised_entities: Also output the normalised label, one row per distinct
            normalised label of an input row.
        include_entity_source: Output `{entityId, entitySource}` structs rather than bare ids.

    Returns:
        The input with the identifiers found added as a list column, null when there are none.
        Identical input rows come out once.

    Raises:
        ValueError: When not exactly one of `type_col_name` and `type_col` is given, the kind is
            unknown, or an output column already exists.
    """
    if type_col_name is not None and type_col is None:
        entity_type = pl.col(type_col_name)
    elif type_col is not None and type_col_name is None:
        entity_type = type_col
    else:
        raise ValueError("Exactly one of 'type_col_name' or 'type_col' must be provided.")
    if entity_kind not in {'label', 'id'}:
        raise ValueError(f"entity_kind must be 'label' or 'id', not {entity_kind!r}")

    columns = lf.collect_schema().names()
    extra_columns = [NORMALISED_COLUMN] if include_normalised_entities else []
    if clashing := {result_col_name, *extra_columns} & set(columns):
        raise ValueError(f'output columns already present in the input: {sorted(clashing)}')

    indexed = lf.with_row_index(_ROW)
    query = indexed.select(_ROW, pl.col(entity_col_name).alias(_ENTITY), entity_type.alias(_TYPE))
    query = _query_labels(query) if entity_kind == 'label' else _query_ids(query)

    lut = lut.filter(pl.col('entityKind') == entity_kind).select(
        NORMALISED_COLUMN,
        pl.col('entityType').alias(_TYPE),
        'entityIds',
    )
    mapped = _normalise_query(query).join(lut, on=[NORMALISED_COLUMN, _TYPE], how='left')

    ids = pl.col('entityIds')
    if not include_entity_source:
        ids = ids.list.eval(pl.element().struct.field('entityId'))
    found = (
        mapped
        .group_by(_ROW, *extra_columns)
        .agg(ids.list.explode(keep_nulls=False, empty_as_null=False).unique().sort().alias(result_col_name))
        .with_columns(
            pl.when(pl.col(result_col_name).list.len() > 0).then(pl.col(result_col_name)).alias(result_col_name)
        )
    )
    return (
        indexed
        .join(found, on=_ROW, how='left', maintain_order='left')
        .select(*columns, *extra_columns, result_col_name)
        .unique(maintain_order=True)
    )
