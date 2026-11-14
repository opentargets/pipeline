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


def association_otf_schema() -> pl.Schema:
    """The association_otf output: one row per evidence with its disease and target labels and novelties."""
    return pl.Schema({
        'rowId': pl.String(),
        'diseaseId': pl.String(),
        'targetId': pl.String(),
        'datasourceId': pl.String(),
        'datatypeId': pl.String(),
        'rowScore': pl.Float64(),
        'diseaseData': pl.String(),
        'targetData': pl.String(),
        'noveltyDirect': pl.Float64(),
        'noveltyIndirect': pl.Float64(),
    })


def association_timeseries_schema() -> pl.Schema:
    """The association_timeseries_view output: one row per association, aggregation value and year."""
    return pl.Schema({
        'diseaseId': pl.String(),
        'targetId': pl.String(),
        'aggregationType': pl.String(),
        'aggregationValue': pl.String(),
        'year': pl.Int32(),
        'associationScore': pl.Float64(),
        'novelty': pl.Float64(),
        'yearlyEvidenceCount': pl.Int64(),
        'isDirect': pl.Boolean(),
        'partitionKey': pl.String(),
    })
