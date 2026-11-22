"""Schemas for the association step."""

import polars as pl


def association_schema(evidence_count_dtype: pl.DataType) -> pl.Schema:
    """The association output: one row per target, disease and aggregation value, with its yearly timeseries."""
    return pl.Schema({
        'diseaseId': pl.String(),
        'targetId': pl.String(),
        'aggregationType': pl.String(),
        'aggregationValue': pl.String(),
        'associationScore': pl.Float64(),
        'evidenceCount': pl.Int64(),
        'timeseries': pl.List(
            pl.Struct({
                'year': pl.Int32(),
                'associationScore': pl.Float64(),
                'novelty': pl.Float64(),
                'yearlyEvidenceCount': evidence_count_dtype,
            })
        ),
        'currentNovelty': pl.Float64(),
    })
