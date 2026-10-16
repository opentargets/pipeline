"""Schema for the gene sets dataset."""

import polars as pl

gene_sets_schema = {
    'id': pl.String,
    'label': pl.String,
    'source': pl.String,
    'category': pl.String,
    'parentIds': pl.List(pl.String),
    'entityIds': pl.List(pl.String),
}
