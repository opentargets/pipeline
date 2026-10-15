"""Schema for amino-acid variant annotations."""

import polars as pl

from pts.schemas.variant_annotation import variant_effect_dtype

amino_acid_variants_schema = pl.Schema({
    'uniprotAccession': pl.String,
    'aminoAcidChange': pl.String,
    'variantEffect': variant_effect_dtype,
})
