"""Schemas for the target tractability step."""

import polars as pl

tractability_schema = pl.Schema({
    'targetId': pl.String(),
    'modality': pl.String(),
    'category': pl.String(),
    'value': pl.Boolean(),
})
"""The tractability index: one row per target, modality and category."""
