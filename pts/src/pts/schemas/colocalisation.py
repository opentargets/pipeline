"""Schemas of the colocalisation step: the input columns it reads and the output it writes."""

from __future__ import annotations

import polars as pl

COLOC_PIP_ECAVIAR = 'COLOC_PIP_ECAVIAR'
"""The colocalisation method name written to `colocalisationMethod`."""

CREDIBLE_SET_COLUMNS = ('id', 'studyType', 'chromosome')
"""Columns read from the credible sets."""

CREDIBLE_SET_VARIANT_COLUMNS = ('credibleSetId', 'variantId', 'posteriorProbability', 'conditionalBeta', 'reportedBeta')
"""Columns read from the credible set variants."""

colocalisation_schema = pl.Schema({
    'leftLocusId': pl.String(),
    'rightLocusId': pl.String(),
    'rightStudyType': pl.String(),
    'chromosome': pl.String(),
    'colocalisationMethod': pl.String(),
    'numberColocalisingVariants': pl.Int64(),
    'h3': pl.Float64(),
    'h4': pl.Float64(),
    'clpp': pl.Float64(),
    'betaRatioSignAverage': pl.Float64(),
})
"""The colocalisation output: one row per overlapping pair of credible sets."""
