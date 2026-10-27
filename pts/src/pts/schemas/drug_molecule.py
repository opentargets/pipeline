"""Schemas for the drug molecule index."""

import polars as pl

label_source_schema = pl.List(pl.Struct({'label': pl.String(), 'source': pl.String()}))

cross_reference_schema = pl.List(pl.Struct({'source': pl.String(), 'ids': pl.List(pl.String())}))

molecule_schema = pl.Schema({
    'id': pl.String(),
    'canonicalSmiles': pl.String(),
    'inchiKey': pl.String(),
    'molblock': pl.String(),
    'drugType': pl.String(),
    'name': pl.String(),
    'parentId': pl.String(),
    'synonyms': label_source_schema,
    'tradeNames': label_source_schema,
    'crossReferences': cross_reference_schema,
    'childChemblIds': pl.List(pl.String()),
})
"""The chembl_molecule intermediate, as the drug_molecule step consumes it."""

structural_alert_schema = pl.List(pl.Struct({'alertId': pl.Int64(), 'alertName': pl.String(), 'setName': pl.String()}))

molecule_properties_schema = pl.Schema({
    'mwFreebase': pl.Float64(),
    'alogp': pl.Float64(),
    'aromaticRings': pl.Int32(),
    'heavyAtoms': pl.Int32(),
    'qedWeighted': pl.Float64(),
    'numRo5Violations': pl.Int32(),
    'structuralAlerts': structural_alert_schema,
    'structureType': pl.String(),
    'naturalProduct': pl.Boolean(),
    'moleculeClass': pl.String(),
})
"""Compound properties the chembl_molecule intermediate adds for every molecule; not in the index."""

drug_molecule_schema = pl.Schema({
    **molecule_schema,
    'maximumClinicalStage': pl.String(),
    'description': pl.String(),
})
"""The index: every molecule column passed through, plus the two derived here."""
