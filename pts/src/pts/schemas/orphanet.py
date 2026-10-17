"""Schemas used in the orphanet evidence."""

import polars as pl

#: The rows `parse_orphanet_xml` yields, one per disorder-gene association.
ORPHANET_SCHEMA = pl.Schema({
    'diseaseFromSource': pl.String,
    'diseaseFromSourceId': pl.String,
    'type': pl.String,
    'literature': pl.List(pl.String),
    'associationType': pl.String,
    'confidence': pl.String,
    'targetFromSource': pl.String,
    'targetFromSourceId': pl.String,
})
