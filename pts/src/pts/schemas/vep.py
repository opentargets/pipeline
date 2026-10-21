"""VEP dataset schema."""

import polars as pl

vep_schema = pl.Schema({
    'input': pl.String,
    'most_severe_consequence': pl.String,
    'intergenic_consequences': pl.List(pl.Struct({'hgvsg': pl.String})),
    'colocated_variants': pl.List(
        pl.Struct({
            'id': pl.String,
            'var_synonyms': pl.Struct({'ClinVar': pl.List(pl.String), 'OMIM': pl.List(pl.String)}),
        })
    ),
    'transcript_consequences': pl.List(
        pl.Struct({
            'conservation': pl.Float64,
            'alphamissense': pl.Struct({'am_class': pl.String, 'am_pathogenicity': pl.Float64}),
            'amino_acids': pl.String,
            'hgvsg': pl.String,
            'gene_symbol': pl.String,
            'biotype': pl.String,
            'appris': pl.String,
            'mane_select': pl.String,
            'tssdistance': pl.Int64,
            'cadd_phred': pl.Float64,
            'canonical': pl.Int64,
            'codons': pl.String,
            'consequence_terms': pl.List(pl.String),
            'distance': pl.Int64,
            'gene_id': pl.String,
            'impact': pl.String,
            'lof': pl.String,
            'lof_filter': pl.String,
            'protein_end': pl.Int64,
            'swissprot': pl.List(pl.String),
            'transcript_id': pl.String,
            'trembl': pl.List(pl.String),
        })
    ),
})
"""The VEP JSON fields the variant index reads; the rest of each record is skipped."""
