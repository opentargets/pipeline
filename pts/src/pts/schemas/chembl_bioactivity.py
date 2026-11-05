"""Schema for the ChEMBL bioactivity dataset."""

import polars as pl

drug_fields = {
    'drugFromSourceId': pl.String(),
    'drugId': pl.String(),
}

document_fields = {
    'documentId': pl.String(),
    'docType': pl.String(),
    'pubmedId': pl.Int64(),
    'year': pl.Int32(),
}

chembl_bioactivity_schema = pl.Schema({
    'activityId': pl.Int64(),
    'targetId': pl.String(),
    **drug_fields,
    'pchemblValue': pl.Float64(),
    'standardMeasurementType': pl.String(),
    'standardMeasurementValue': pl.Float64(),
    'measurementClass': pl.String(),
    'activityOutcome': pl.String(),
    'assayActionType': pl.String(),
    'actionTypes': pl.List(pl.String()),
    'le': pl.Float64(),
    'lle': pl.Float64(),
    'assayId': pl.String(),
    'assayType': pl.String(),
    'assayDescription': pl.String(),
    'assayOrganism': pl.String(),
    'confidenceScore': pl.Int16(),
    'variantFromSource': pl.String(),
    'targetFromSourceId': pl.String(),
    'complexFlag': pl.Boolean(),
    **document_fields,
})
"""output/chembl_bioactivity: one row per non-degradation ChEMBL activity with a pChEMBL value and
Open Targets target it credits, keyed by (activityId, targetId)."""
