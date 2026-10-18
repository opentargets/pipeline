"""Schemas of OnToma's lookup tables."""

import polars as pl

#: OnToma's `RawEntityLUT`: one entity label or id from one source, before normalisation.
RAW_ENTITY_LUT = pl.Schema({
    'entityId': pl.String,
    'entityLabel': pl.String,
    'entityScore': pl.Float64,
    'nlpPipelineTrack': pl.String,
    'entitySource': pl.String,
    'entityType': pl.String,
    'entityKind': pl.String,
})

#: OnToma's `EntityLUT`: the ids a normalised label or id maps to.
ENTITY_LUT = pl.Schema({
    'entityKind': pl.String,
    'entityType': pl.String,
    'entityLabelNormalised': pl.String,
    'entityIds': pl.List(pl.Struct({'entityId': pl.String, 'entitySource': pl.String})),
})
ENTITY_LUT_KEY = ('entityKind', 'entityType', 'entityLabelNormalised')
