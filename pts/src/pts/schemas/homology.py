"""Schemas for the homology step."""

import polars as pl

orthologs_schema = pl.Schema({
    'id': pl.String(),
    'speciesId': pl.String(),
    'speciesName': pl.String(),
    'homologyType': pl.String(),
    'targetGeneId': pl.String(),
    'isHighConfidence': pl.String(),
    'targetGeneSymbol': pl.String(),
    'queryPercentageIdentity': pl.Float64(),
    'targetPercentageIdentity': pl.Float64(),
    'priority': pl.Int32(),
})
"""Homologue pairs keyed by the human query gene `id`, before it is resolved against output/target."""

homology_schema = pl.Schema({
    'targetId': pl.String(),
    'speciesId': pl.String(),
    'speciesName': pl.String(),
    'homologyType': pl.String(),
    'targetGeneId': pl.String(),
    'isHighConfidence': pl.String(),
    'targetGeneSymbol': pl.String(),
    'queryPercentageIdentity': pl.Float64(),
    'targetPercentageIdentity': pl.Float64(),
    'priority': pl.Int32(),
})
"""The homology dataset: one row per target and homologue."""
