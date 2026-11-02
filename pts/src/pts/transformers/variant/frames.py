"""The contract between the stages of a chunk: the frames `chunk_frames` builds and `index_rows` reads."""

from __future__ import annotations

from typing import NamedTuple

import polars as pl


class ChunkFrames(NamedTuple):
    """The annotation of one chunk: a row per variant and a row per canonical transcript consequence."""

    variants: pl.DataFrame
    consequences: pl.DataFrame


FIELDS = pl.Schema({
    'transcript_id': pl.String,
    'gene_id': pl.String,
    'gene_symbol': pl.String,
    'biotype': pl.String,
    'appris': pl.String,
    'mane_select': pl.String,
    'tssdistance': pl.Int64,
    'cds_end': pl.Int64,
    'codons': pl.String,
    'consequence_terms': pl.List(pl.String),
    'distance': pl.Int64,
    'impact': pl.String,
    'amino_acids': pl.String,
    'protein_end': pl.Int64,
    'swissprot': pl.List(pl.String),
    'trembl': pl.List(pl.String),
})
"""The transcript fields of the consequences frame, between the copied variant columns and `rank`."""

VARIANT_COLUMNS = (
    'row',
    'chromosome',
    'position',
    'referenceAllele',
    'alternateAllele',
    'vepStart',
    'vepEnd',
    'vepReference',
    'vepAlternate',
    'hgvsg',
)
"""The variants columns that precede the severity columns `chunk_frames` computes."""

_SEVERITY = pl.Schema({'most_severe_consequence': pl.String, 'reported_transcripts': pl.Boolean})
_VARIANT_DTYPES = {
    'row': pl.Int64,
    'chromosome': pl.String,
    'position': pl.Int32,
    'referenceAllele': pl.String,
    'alternateAllele': pl.String,
    'vepStart': pl.Int64,
    'vepEnd': pl.Int64,
    'vepReference': pl.String,
    'vepAlternate': pl.String,
    'hgvsg': pl.String,
}
VARIANTS_SCHEMA = pl.Schema({name: _VARIANT_DTYPES[name] for name in VARIANT_COLUMNS} | _SEVERITY)
"""The columns of the variants frame, before `add_colocated` adds its id lists."""

COPIED = (
    'row',
    'position',
    'referenceAllele',
    'alternateAllele',
    'vepStart',
    'vepEnd',
    'vepReference',
    'vepAlternate',
)
"""The variants columns each consequence repeats."""

CONSEQUENCES_SCHEMA = pl.Schema({
    **{name: VARIANTS_SCHEMA[name] for name in COPIED},
    **FIELDS,
    'rank': pl.Int32,
})
"""The columns of the consequences frame, before any effect adds its own."""

EFFECTS_SCHEMA = pl.Schema({
    'conservation': pl.Float64,
    'am_class': pl.String,
    'am_pathogenicity': pl.Float64,
    'lof': pl.String,
    'lof_filter': pl.String,
    'sift_score': pl.Float64,
    'sift_prediction': pl.String,
})
"""The columns the effects add to the consequences frame."""

ID_LISTS = pl.Schema({'rsIds': pl.List(pl.String), 'clinvarIds': pl.List(pl.String), 'omimIds': pl.List(pl.String)})
"""The identifiers of each row's colocated variants, as `colocated_variants` returns them."""


def require(frame: pl.DataFrame, schema: pl.Schema, name: str) -> None:
    """Check that `frame` has every column name of `schema`; dtypes are not checked.

    Raises:
        ValueError: naming the `name` frame and its missing columns.
    """
    present = set(frame.columns)
    missing = [column for column in schema.names() if column not in present]
    if missing:
        raise ValueError(f'{name} frame lacks columns: {", ".join(missing)}')
