"""Schema for the Reactome pathway graph dataset."""

import polars as pl

reactome_schema = {
    'id': pl.String,
    'label': pl.String,
    'ancestors': pl.List(pl.String),
    'descendants': pl.List(pl.String),
    'children': pl.List(pl.String),
    'parents': pl.List(pl.String),
    'path': pl.List(pl.List(pl.String)),
}
