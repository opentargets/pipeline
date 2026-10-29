"""Schema for the ChEMBL bioactivity dataset."""

import polars as pl

target_component_fields = {
    'uniprotId': pl.String(),
    'targetId': pl.String(),
    'approvedSymbol': pl.String(),
    'inBindingSite': pl.Boolean(),
}

target_fields = {
    'targetChemblId': pl.String(),
    'targetType': pl.String(),
    'targetName': pl.String(),
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
    'moleculeId': pl.String(),
    'pchemblValue': pl.Float64(),
    'standardType': pl.String(),
    'standardValue': pl.Float64(),
    'standardUnits': pl.String(),
    'measurementClass': pl.String(),
    'actionType': pl.String(),
    'le': pl.Float64(),
    'lle': pl.Float64(),
    'assayId': pl.String(),
    'assayType': pl.String(),
    'assayDescription': pl.String(),
    'assayOrganism': pl.String(),
    'confidenceScore': pl.Int16(),
    'assayVariantMutation': pl.String(),
    **target_fields,
    'targetComponents': pl.List(pl.Struct(target_component_fields)),
    'complexFlag': pl.Boolean(),
    **document_fields,
})
"""output/chembl_bioactivity: one row per non-degradation ChEMBL activity with a pChEMBL value."""
