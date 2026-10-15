"""VARIANT dataset schema."""

import polars as pl

from pts.schemas.amino_acid_variants import variant_effect_dtype

variant_schema = pl.Schema({
    'variantId': pl.String,
    'chromosome': pl.String,
    'position': pl.Int32,
    'referenceAllele': pl.String,
    'alternateAllele': pl.String,
    'variantEffect': variant_effect_dtype,
    'mostSevereConsequenceId': pl.String,
    'transcriptConsequences': pl.List(
        pl.Struct({
            'variantFunctionalConsequenceIds': pl.List(pl.String),
            'aminoAcidChange': pl.String,
            'uniprotAccessions': pl.List(pl.String),
            'isEnsemblCanonical': pl.Boolean,
            'codons': pl.String,
            'distanceFromFootprint': pl.Int64,
            'distanceFromTss': pl.Int64,
            'appris': pl.String,
            'maneSelect': pl.String,
            'targetId': pl.String,
            'impact': pl.String,
            'lofteePrediction': pl.String,
            'siftPrediction': pl.Float32,
            'polyphenPrediction': pl.Float32,
            'consequenceScore': pl.Float32,
            'transcriptIndex': pl.Int32,
            'approvedSymbol': pl.String,
            'biotype': pl.String,
            'transcriptId': pl.String,
        })
    ),
    'rsIds': pl.List(pl.String),
    'hgvsId': pl.String,
    'alleleFrequencies': pl.List(pl.Struct({'populationName': pl.String, 'alleleFrequency': pl.Float64})),
    'dbXrefs': pl.List(pl.Struct({'id': pl.String, 'source': pl.String})),
    'variantDescription': pl.String,
})
