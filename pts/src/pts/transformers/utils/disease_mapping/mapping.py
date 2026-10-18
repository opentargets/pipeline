"""Polars port of `OnToma.map_entities`: resolve labels and ids against an OnToma lookup table.

The lookup tables themselves are still built by OnToma on spark (`pyspark ontoma_lut_generation`);
only the query side is ported. It must normalise exactly as the tables were, so it is tied to the
spark-nlp version that built them -- see `pts.transformers.utils.disease_mapping.normalise`.

Behaviour kept from OnToma on purpose, because it is observable in the output:

* **Identical input rows collapse into one.** OnToma aggregates by every input column, so
  duplicates do not survive the mapping. This port deduplicates the same way.
* **Each label is tried in two spellings**, translated with and without accented letters, and
  the identifiers found for either are merged.
* **No identifier means null, not an empty list.**

Differences, none of which can change which identifiers are found for the current callers:

* The identifier lists are sorted. OnToma's come out of `collect_set` in no particular order.
* Rows keep their input order, which spark's aggregation does not.
* Input columns are never touched. OnToma works in the input frame itself, so input columns named
  like its working columns (`entityLabel`, `entityType`, `entityKind`, `nlpPipelineTrack`) are
  overwritten, and an `entityLabel` label column even changes which track a label takes. Here the
  working columns carry a random prefix that no input column has, and are dropped before
  returning; only an output column already in the input is refused.

Diffed against `OnToma.map_entities` on spark with lookup tables built by OnToma from the disease
index: 162,085 query rows covering names, synonyms, cross references and perturbed copies of
them, mapped as labels and as ids, with a type column of mixed and null types and with both
`include_*` flags, matched exactly in every configuration.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Literal

import polars as pl
import polars.selectors as cs

from pts.transformers.utils.dataset import scan_dataset
from pts.transformers.utils.disease_mapping.normalise import normalise_batch
from pts.transformers.utils.disease_mapping.query import (
    alternative_translations,
    clean_disease_label,
    determine_track,
    format_identifier,
)
from pts.transformers.utils.spark_strings import spark_trim

EntityKind = Literal['label', 'id']

#: Column of the normalised label, in the lookup table and in the output when it is asked for.
NORMALISED_COLUMN = 'entityLabelNormalised'


def temporary_prefix(columns: list[str]) -> str:
    """A random column name prefix no column in `columns` starts with, for working columns."""
    while any(column.startswith(prefix := f'__{uuid.uuid4().hex[:8]}_') for column in columns):
        pass
    return prefix


@dataclass(frozen=True)
class _Working:
    """Names of the working columns of one `map_entities` call, all under one random prefix."""

    prefix: str
    entity: str
    type: str
    label: str
    track: str
    normalised: str
    ids: str

    @classmethod
    def under(cls, prefix: str) -> _Working:
        names = ('entity', 'type', 'label', 'track', 'normalised', 'ids')
        return cls(prefix, *(f'{prefix}{name}' for name in names))


def read_entity_lut(path: str) -> pl.LazyFrame:
    """Read a lookup table OnToma wrote (`ReadyEntityLUT`), as written by its `cache_dir`."""
    return scan_dataset(path).select('entityKind', 'entityType', NORMALISED_COLUMN, 'entityIds')


def _query_labels(query: pl.LazyFrame, w: _Working) -> pl.LazyFrame:
    """`OnToma._extract_query_entity_labels`."""
    entity = pl.col(w.entity)
    entity_type = pl.col(w.type)
    return (
        query
        .with_columns(
            alternative_translations(spark_trim(entity)).alias(w.label),
            # decided on the raw label, before trimming, translating or cleaning
            pl.when(entity_type == 'CD').then(pl.lit('symbol')).otherwise(determine_track(entity)).alias(w.track),
        )
        .explode(w.label)
        .with_columns(
            pl
            .when(entity_type == 'DS')
            .then(clean_disease_label(pl.col(w.label)))
            .otherwise(pl.col(w.label))
            .alias(w.label),
        )
    )


def _query_ids(query: pl.LazyFrame, w: _Working) -> pl.LazyFrame:
    """`OnToma._extract_query_entity_ids`."""
    return query.with_columns(
        format_identifier(spark_trim(pl.col(w.entity)).str.to_uppercase()).alias(w.label),
        pl.lit('symbol').alias(w.track),
    )


def _normalise_query(query: pl.LazyFrame, w: _Working) -> pl.LazyFrame:
    """`OnToma._normalise_entities`: normalise each distinct label once and join the result back.

    The join is an equi-join as in OnToma, so a null label stays unnormalised.
    """
    distinct = (
        query
        .select(w.label, w.track)
        .unique()
        .with_columns(
            pl.map_batches([w.label, w.track], lambda series: normalise_batch(*series), return_dtype=pl.String).alias(
                w.normalised
            ),
        )
    )
    return query.join(distinct, on=[w.label, w.track], how='left')


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
        The input, untouched, with the identifiers found added as a list column, null when there
        are none. Identical input rows come out once.

    Raises:
        ValueError: When not exactly one of `type_col_name` and `type_col` is given, the kind is
            unknown, or an output column is already in the input.
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
    output_columns = [result_col_name, *([NORMALISED_COLUMN] if include_normalised_entities else [])]
    if clashing := set(output_columns) & set(columns):
        raise ValueError(f'output columns already present in the input: {sorted(clashing)}')

    w = _Working.under(temporary_prefix(columns))
    keyed = lf.with_columns(pl.col(entity_col_name).alias(w.entity), entity_type.alias(w.type))
    keys = [w.entity, w.type]
    query = keyed.select(keys).unique()
    query = _query_labels(query, w) if entity_kind == 'label' else _query_ids(query, w)

    lut = lut.filter(pl.col('entityKind') == entity_kind).select(
        pl.col(NORMALISED_COLUMN).alias(w.normalised),
        pl.col('entityType').alias(w.type),
        pl.col('entityIds').alias(w.ids),
    )
    mapped = _normalise_query(query, w).join(lut, on=[w.normalised, w.type], how='left')

    ids = pl.col(w.ids)
    if not include_entity_source:
        ids = ids.list.eval(pl.element().struct.field('entityId'))
    found = (
        mapped
        .group_by(*keys, *([w.normalised] if include_normalised_entities else []))
        .agg(ids.list.explode(keep_nulls=False, empty_as_null=False).unique().sort().alias(result_col_name))
        .with_columns(
            pl.when(pl.col(result_col_name).list.len() > 0).then(pl.col(result_col_name)).alias(result_col_name)
        )
        .rename({w.normalised: NORMALISED_COLUMN}, strict=False)
    )
    return (
        keyed
        .join(found, on=keys, how='left', nulls_equal=True, maintain_order='left')
        # leaves the input columns, then the normalised label when asked for, then the result
        .drop(cs.starts_with(w.prefix))
        .unique(maintain_order=True)
    )
