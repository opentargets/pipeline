"""Schema for the disease location dataset."""

import polars as pl

disease_location_schema = pl.Schema({
    'diseaseId': pl.String(),
    'biosampleId': pl.String(),
})
"""One row per disease and its most specific anatomical location."""
