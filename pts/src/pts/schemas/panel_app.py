"""Schemas used in the Genomics England PanelApp evidence."""

import polars as pl

#: The PanelApp TSV columns the step reads, typed as spark's `inferSchema` reads them.
PANEL_APP_INPUT_SCHEMA = pl.Schema({
    'Symbol': pl.String,
    'Panel Id': pl.Int32,
    'Panel Name': pl.String,
    'Panel Version': pl.Float64,
    'Panel Status': pl.String,
    'List': pl.String,
    'Mode of inheritance': pl.String,
    'Phenotypes': pl.String,
})

#: The fields of the PanelApp panels JSON lines the step reads.
PANELS_SCHEMA = pl.Schema({
    'id': pl.Int64,
    'genes': pl.List(
        pl.Struct({
            'gene_data': pl.Struct({'gene_symbol': pl.String}),
            'publications': pl.List(pl.String),
        })
    ),
})
