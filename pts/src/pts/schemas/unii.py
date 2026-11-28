"""Schemas for the FDA UNII annotation of ChEMBL molecules."""

import polars as pl

unii_inn_schema = pl.Schema({
    'unii': pl.String(),
    'inn': pl.String(),
    'inchiKey': pl.String(),
})
"""One row per INN name of a UNII substance with an INN number; `inchiKey` is null without a structure."""

unii_annotation_schema = pl.Schema({
    'inchiKey': pl.String(),
    'unii': pl.String(),
    'inn': pl.String(),
    'synonyms': pl.List(pl.String()),
    'tradeNames': pl.List(pl.String()),
    'ncit': pl.String(),
})
"""One row per InChIKey of a single INN substance: the INN, official names and codes, brands and NCIt id."""
