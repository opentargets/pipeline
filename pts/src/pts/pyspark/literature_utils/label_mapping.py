"""Map literature match labels to entity ids with the polars disease mapping, as `OnToma.map_entities` did."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pyarrow as pa

from pts.transformers.utils.disease_mapping import map_entities, read_entity_lut

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

#: `map_label_batches` output, as a spark DDL.
MAPPED_LABELS_DDL = (
    'label string, type string, entityLabelNormalised string, '
    'entityIds array<struct<entityId: string, entitySource: string>>'
)

_MAPPED_LABELS = pa.schema([
    ('label', pa.string()),
    ('type', pa.string()),
    ('entityLabelNormalised', pa.string()),
    ('entityIds', pa.list_(pa.struct([('entityId', pa.string()), ('entitySource', pa.string())]))),
])

_LOOKUP_TABLES: dict[str, pl.DataFrame] = {}


def lookup_table(path: str) -> pl.DataFrame:
    """The label rows of the lookup table at `path`, read once per process."""
    if path not in _LOOKUP_TABLES:
        _LOOKUP_TABLES[path] = read_entity_lut(path).filter(pl.col('entityKind') == 'label').collect()
    return _LOOKUP_TABLES[path]


def map_distinct_labels(labels: pl.DataFrame, lut: pl.DataFrame) -> pl.DataFrame:
    """Map distinct (`label`, `type`) pairs: a row per pair and normalised spelling, `entityIds` null if unmapped."""
    return map_entities(
        labels.lazy(),
        lut.lazy(),
        result_col_name='entityIds',
        entity_col_name='label',
        entity_kind='label',
        type_col_name='type',
        include_normalised_entities=True,
        include_entity_source=True,
    ).collect()


def map_label_batches(lut_path: str) -> Callable[[Iterator[pa.RecordBatch]], Iterator[pa.RecordBatch]]:
    """A `mapInArrow` function mapping batches of (`label`, `type`) pairs against the lookup table at `lut_path`."""

    def map_batches(batches: Iterator[pa.RecordBatch]) -> Iterator[pa.RecordBatch]:
        lut = lookup_table(lut_path)
        for batch in batches:
            labels = pl.from_arrow(batch, schema={'label': pl.String, 'type': pl.String})
            yield from map_distinct_labels(labels, lut).to_arrow().cast(_MAPPED_LABELS).to_batches()

    return map_batches
