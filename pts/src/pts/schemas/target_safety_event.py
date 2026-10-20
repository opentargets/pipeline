"""Schemas for the target safety event step."""

import polars as pl

effect = pl.Struct({'direction': pl.String(), 'dosing': pl.String()})

biosample = pl.Struct({
    'tissueLabel': pl.String(),
    'tissueId': pl.String(),
    'cellLabel': pl.String(),
    'cellFormat': pl.String(),
    'cellId': pl.String(),
})

study = pl.Struct({'description': pl.String(), 'name': pl.String(), 'type': pl.String()})

aopwiki_schema = pl.Schema({
    'id': pl.String(),
    'event': pl.String(),
    'eventId': pl.String(),
    'datasource': pl.String(),
    'url': pl.String(),
    'effects': pl.List(effect),
    'biosamples': pl.List(biosample),
})
"""The AOPWiki input."""

brennan_schema = pl.Schema({
    'id': pl.String(),
    'event': pl.String(),
    'eventId': pl.String(),
    'datasource': pl.String(),
    'literature': pl.String(),
    'effects': effect,
    'studies': study,
    'biosamples': biosample,
})
"""The Brennan secondary pharmacology input."""

safety_evidence_schema = pl.Schema({
    'id': pl.String(),
    'targetFromSourceId': pl.String(),
    'event': pl.String(),
    'eventId': pl.String(),
    'datasource': pl.String(),
    'effects': pl.List(effect),
    'literature': pl.String(),
    'url': pl.String(),
    'biosample': biosample,
    'study': study,
    'supporting_variation': pl.String(),
})
"""Every source harmonised to one shape, before target resolution."""
