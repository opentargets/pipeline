"""Schema for the ChEMBL bioactivity dataset."""

import polars as pl

target_component_fields = {
    'uniprotId': pl.String(),
    'targetId': pl.String(),
    'inBindingSite': pl.Boolean(),
}

drug_fields = {
    'drugFromSourceId': pl.String(),
    'drugId': pl.String(),
}

document_fields = {
    'documentId': pl.String(),
    'docType': pl.String(),
    'pubmedId': pl.Int64(),
    'doi': pl.String(),
    'year': pl.Int32(),
}

chembl_bioactivity_schema = pl.Schema({
    'activityId': pl.Int64(),
    **drug_fields,
    'pchemblValue': pl.Float64(),
    'standardType': pl.String(),
    'standardValue': pl.Float64(),
    'standardUnits': pl.String(),
    'measurementClass': pl.String(),
    'assayActionType': pl.String(),
    'actionTypes': pl.List(pl.String()),
    'le': pl.Float64(),
    'lle': pl.Float64(),
    'assayId': pl.String(),
    'assayType': pl.String(),
    'assayDescription': pl.String(),
    'assayOrganism': pl.String(),
    'confidenceScore': pl.Int16(),
    'assayVariantMutation': pl.String(),
    'targetFromSourceId': pl.String(),
    'targetComponents': pl.List(pl.Struct(target_component_fields)),
    'complexFlag': pl.Boolean(),
    **document_fields,
})
"""output/chembl_bioactivity: one row per non-degradation ChEMBL activity with a pChEMBL value."""
