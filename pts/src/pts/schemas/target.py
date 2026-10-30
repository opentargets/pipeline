"""Target step schemas."""

import polars as pl

ID_SOURCE = pl.Struct({'id': pl.String, 'source': pl.String})
LABEL_SOURCE = pl.Struct({'label': pl.String, 'source': pl.String})
LOCATION = pl.Struct({
    'location': pl.String,
    'source': pl.String,
    'termSL': pl.String,
    'labelSL': pl.String,
    'targetModifier': pl.String,
})
GENOMIC_LOCATION = pl.Struct({'chromosome': pl.String, 'start': pl.Int64, 'end': pl.Int64, 'strand': pl.Int32})
CANONICAL_TRANSCRIPT = pl.Struct({
    'id': pl.String,
    'chromosome': pl.String,
    'start': pl.Int64,
    'end': pl.Int64,
    'strand': pl.Int32,
})
GO = pl.Struct({
    'id': pl.String,
    'source': pl.String,
    'evidence': pl.String,
    'aspect': pl.String,
    'geneProduct': pl.String,
    'ecoId': pl.String,
})
HALLMARK_ATTRIBUTE = pl.Struct({'pmid': pl.Int64, 'description': pl.String, 'name': pl.String})
CANCER_HALLMARK = pl.Struct({'pmid': pl.Int64, 'description': pl.String, 'impact': pl.String, 'label': pl.String})
HALLMARKS = pl.Struct({'attributes': pl.List(HALLMARK_ATTRIBUTE), 'cancerHallmarks': pl.List(CANCER_HALLMARK)})
TARGET_CLASS = pl.Struct({'id': pl.Int64, 'label': pl.String, 'level': pl.String})
CONSTRAINT = pl.Struct({
    'constraintType': pl.String,
    'score': pl.Float32,
    'exp': pl.Float32,
    'obs': pl.Int32,
    'oe': pl.Float32,
    'oeLower': pl.Float32,
    'oeUpper': pl.Float32,
    'upperRank': pl.Int32,
    'upperBin': pl.Int32,
    'upperBin6': pl.Int32,
})
PATHWAY = pl.Struct({'pathwayId': pl.String, 'pathway': pl.String, 'topLevelTerm': pl.String})

HGNC_DOC = pl.Schema({
    'ensembl_gene_id': pl.String,
    'hgnc_id': pl.String,
    'symbol': pl.String,
    'name': pl.String,
    'uniprot_ids': pl.List(pl.String),
    'alias_symbol': pl.List(pl.String),
    'alias_name': pl.List(pl.String),
    'prev_symbol': pl.List(pl.String),
    'prev_name': pl.List(pl.String),
})

UNIPROT_ENTRY = pl.Schema({
    'id': pl.String,
    'reviewed': pl.Boolean,
    'accessions': pl.List(pl.String),
    'names': pl.List(pl.String),
    'synonyms': pl.List(pl.String),
    'symbolSynonyms': pl.List(pl.String),
    'dbXrefs': pl.List(pl.String),
    'functions': pl.List(pl.String),
    'locations': pl.List(pl.Struct({'location': pl.String, 'targetModifier': pl.String})),
})

TARGET = pl.Schema({
    'id': pl.String,
    'approvedSymbol': pl.String,
    'biotype': pl.String,
    'genomicLocation': GENOMIC_LOCATION,
    'approvedName': pl.String,
    'go': pl.List(GO),
    'hallmarks': HALLMARKS,
    'synonyms': pl.List(LABEL_SOURCE),
    'symbolSynonyms': pl.List(LABEL_SOURCE),
    'nameSynonyms': pl.List(LABEL_SOURCE),
    'functionDescriptions': pl.List(pl.String),
    'subcellularLocations': pl.List(LOCATION),
    'targetClass': pl.List(TARGET_CLASS),
    'obsoleteSymbols': pl.List(LABEL_SOURCE),
    'obsoleteNames': pl.List(LABEL_SOURCE),
    'constraint': pl.List(CONSTRAINT),
    'proteinIds': pl.List(ID_SOURCE),
    'dbXrefs': pl.List(ID_SOURCE),
    'pathways': pl.List(PATHWAY),
    'tss': pl.Int64,
})
