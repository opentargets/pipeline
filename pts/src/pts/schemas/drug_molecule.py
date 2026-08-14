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

drug_molecule_schema = pl.Schema({
    **molecule_schema,
    'maximumClinicalStage': pl.String(),
    'description': pl.String(),
})
"""The index: every molecule column passed through, plus the two derived here."""
