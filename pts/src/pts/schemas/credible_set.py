"""Schemas for the CredibleSet and CredibleSetVariant datasets."""

from __future__ import annotations

import polars as pl

from pts.schemas.study import StudyType
from pts.transformers.credible_set.enums import Chromosome, FinemappingMethod

credible_set_schema = pl.Schema({
    'id': pl.String(),
    'chromosome': pl.String(),
    'locusStart': pl.Int32(),
    'locusEnd': pl.Int32(),
    'studyId': pl.String(),
    'studyType': pl.String(),
    'leadVariantId': pl.String(),
    'fineMappingMethod': pl.String(),
    'logBayesFactor': pl.Float64(),
    'confidence': pl.Int8(),
    'purityMeanR2': pl.Float64(),
    'purityMinR2': pl.Float64(),
    'qualityControls': pl.List(pl.String()),
})

validated_credible_set_schema = pl.Schema({
    **credible_set_schema,
    'chromosome': pl.Enum(Chromosome),
    'studyType': pl.Enum(StudyType),
    'fineMappingMethod': pl.Enum(FinemappingMethod),
})
"""The CredibleSet schema of the validated and published sets, with the columns of a fixed vocabulary as enums.

Spark reads the enum columns as strings, so the published dataset keeps the CredibleSet schema.
"""

credible_set_variant_schema = pl.Schema({
    'credibleSetId': pl.String(),
    'variantId': pl.String(),
    'pValueMantissa': pl.Float32(),
    'pValueExponent': pl.Int32(),
    'effectAlleleFrequency': pl.Float32(),
    'sampleSize': pl.Int32(),
    'reportedBeta': pl.Float64(),
    'reportedStandardError': pl.Float64(),
    'posteriorProbability': pl.Float32(),
    'logBayesFactor': pl.Float32(),
    'r2Overall': pl.Float32(),
    'conditionalBeta': pl.Float32(),
    'conditionalStandardError': pl.Float32(),
    'rescaledAbsBeta': pl.Float32(),
    'rescaledStandardError': pl.Float32(),
    'rescaledDirection': pl.Int8(),
    'qualityControls': pl.List(pl.String()),
})

credible_set_required = ('id', 'studyId', 'chromosome', 'locusStart', 'locusEnd', 'leadVariantId', 'fineMappingMethod')
"""The CredibleSet columns that must not be null: the id and the columns it is hashed from."""

credible_set_variant_required = ('credibleSetId', 'variantId', 'posteriorProbability')
"""The CredibleSetVariant columns that must not be null: the keys of a row and its posterior probability."""
