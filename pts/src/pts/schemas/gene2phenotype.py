"""Schemas used in the gene2phenotype evidence."""

import polars as pl

#: The panel files' columns by position; their header names are not read.
G2P_INPUT_SCHEMA = pl.Schema({
    'g2p id': pl.String,
    'gene symbol': pl.String,
    'gene mim': pl.Int32,
    'hgnc id': pl.Int32,
    'previous gene symbols': pl.String,
    'disease name': pl.String,
    'disease mim': pl.String,
    'disease MONDO': pl.String,
    'allelic requirement': pl.String,
    'cross cutting modifier': pl.String,
    'confidence': pl.String,
    'variant consequence': pl.String,
    'variant types': pl.String,
    'molecular mechanism': pl.String,
    'molecular mechanism support': pl.String,
    'molecular mechanism categorisation': pl.String,
    'molecular mechanism evidence': pl.String,
    'phenotypes': pl.String,
    'publications': pl.String,
    'additional mined publications': pl.String,
    'panel': pl.String,
    'comments': pl.String,
    'date of last review': pl.String,
    'review': pl.String,
})
