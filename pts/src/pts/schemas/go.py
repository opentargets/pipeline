"""Schema for Gene Ontology (GO) dataset."""

import polars as pl

go_schema = {
    'id': pl.String,
    'label': pl.String,
    'namespace': pl.String,
    'altIds': pl.List(pl.String),
    'isA': pl.List(pl.String),
    'partOf': pl.List(pl.String),
    'regulates': pl.List(pl.String),
    'negativelyRegulates': pl.List(pl.String),
    'positivelyRegulates': pl.List(pl.String),
    'isObsolete': pl.Boolean,
}
