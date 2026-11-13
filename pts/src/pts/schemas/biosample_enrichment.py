"""Schema for the BiosampleEnrichment dataset."""

import polars as pl

biosample_enrichment_schema = pl.Schema({
    'diseaseId': pl.String(),
    'studyId': pl.String(),
    'biosampleId': pl.String(),
    'tissueBiosampleId': pl.String(),
    'biosampleDataset': pl.String(),
    'biosampleFromSource': pl.String(),
    'enrichmentApproach': pl.String(),
    'beta': pl.Float64(),
    'betaError': pl.Float64(),
    'pValue': pl.Float64(),
})
"""Release schema, as in the croissant `biosample_enrichment` record set.

Exactly one of `studyId` and `diseaseId` is set: `studyId` for a single study, `diseaseId` for
the studies of one disease pooled together. `tissueBiosampleId` is set only for a tissue x cell
type panel, where `biosampleId` is the cell type and `tissueBiosampleId` the tissue it was
profiled in.
"""

biosample_enrichment_statistics_schema = pl.Schema({
    name: biosample_enrichment_schema[name] for name in ('biosampleId', 'beta', 'betaError', 'pValue')
})
"""The columns one panel's fit produces per unit, before the unit and biosample are labelled."""
