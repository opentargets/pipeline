"""Schemas used in the IMPC evidence."""

import polars as pl

#: The disease model summary columns the step reads, typed as spark's `inferSchema` reads them.
DISEASE_MODEL_SUMMARY_SCHEMA = pl.Schema({
    'model_id': pl.String,
    'model_genetic_background': pl.String,
    'model_description': pl.String,
    'disease_id': pl.String,
    'disease_term': pl.String,
    'disease_model_avg_norm': pl.Float64,
    'marker_id': pl.String,
})

#: Column names of the headerless MGI phenotype-genotype report, by position.
MGI_PUBMED_COLUMNS = ['_0', '_1', '_2', 'mp_id', 'literature', 'targetInModelMgiId']
