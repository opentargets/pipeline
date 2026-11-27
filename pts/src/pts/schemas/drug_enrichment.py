"""Schemas of the drug_enrichment step."""

import polars as pl

#: One row per evidence datasource and clinical phase.
drug_enrichment_schema = pl.Schema({
    'datasourceId': pl.String,
    'clinicalPhase': pl.String,
    'oddsRatio': pl.Float64,
    'oddsRatioCiLow': pl.Float64,
    'oddsRatioCiHigh': pl.Float64,
    'oddsRatioPValue': pl.Float64,
    'relativeSuccess': pl.Float64,
    'relativeSuccessCiLow': pl.Float64,
    'relativeSuccessCiHigh': pl.Float64,
    'relativeSuccessPValue': pl.Float64,
    'unsupportedBelowPhase': pl.Int64,
    'unsupportedReachedPhase': pl.Int64,
    'supportedBelowPhase': pl.Int64,
    'supportedReachedPhase': pl.Int64,
    'supportedAssociations': pl.Int64,
})
