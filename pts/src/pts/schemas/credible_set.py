"""Schema for the CredibleSet dataset."""

import polars as pl

credible_set_schema = pl.Schema({
    'studyLocusId': pl.String(),
    'studyType': pl.String(),
    'variantId': pl.String(),
    'chromosome': pl.String(),
    'position': pl.Int32(),
    'region': pl.String(),
    'studyId': pl.String(),
    'beta': pl.Float64(),
    'zScore': pl.Float64(),
    'pValueMantissa': pl.Float32(),
    'pValueExponent': pl.Int32(),
    'effectAlleleFrequencyFromSource': pl.Float32(),
    'standardError': pl.Float64(),
    'subStudyDescription': pl.String(),
    'qualityControls': pl.List(pl.String()),
    'finemappingMethod': pl.String(),
    'credibleSetIndex': pl.Int32(),
    'credibleSetlog10BF': pl.Float64(),
    'purityMeanR2': pl.Float64(),
    'purityMinR2': pl.Float64(),
    'locusStart': pl.Int32(),
    'locusEnd': pl.Int32(),
    'sampleSize': pl.Int32(),
    'ldSet': pl.List(
        pl.Struct({
            'tagVariantId': pl.String(),
            'r2Overall': pl.Float64(),
        })
    ),
    'locus': pl.List(
        pl.Struct({
            'is95CredibleSet': pl.Boolean(),
            'is99CredibleSet': pl.Boolean(),
            'logBF': pl.Float64(),
            'posteriorProbability': pl.Float64(),
            'variantId': pl.String(),
            'pValueMantissa': pl.Float32(),
            'pValueExponent': pl.Int32(),
            'beta': pl.Float64(),
            'standardError': pl.Float64(),
            'r2Overall': pl.Float64(),
        })
    ),
    'confidence': pl.String(),
    'isTransQtl': pl.Boolean(),
})
