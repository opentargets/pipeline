"""Schema for amino-acid variant annotations."""

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

amino_acid_variants_schema = pl.Schema({
    'uniprotAccession': pl.String,
    'aminoAcidChange': pl.String,
    'variantEffect': variant_effect_dtype,
})
