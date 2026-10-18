"""Schema for curated variant annotations and shared variant-effect records."""

import polars as pl

variant_effect_dtype = pl.List(
    pl.Struct({
        'method': pl.String,
        'assessment': pl.String,
        'score': pl.Float32,
        'assessmentFlag': pl.String,
        'targetId': pl.String,
        'normalisedScore': pl.Float64,
    })
)

variant_annotation_schema = pl.Schema({
    'variantId': pl.String,
    'chromosome': pl.String,
    'position': pl.Int32,
    'referenceAllele': pl.String,
    'alternateAllele': pl.String,
    'variantEffect': variant_effect_dtype,
    'variantDescription': pl.String,
})

amino_acid_variant_schema = pl.Schema({
    'uniprotAccession': pl.String,
    'aminoAcidChange': pl.String,
    'variantEffect': variant_effect_dtype,
})
