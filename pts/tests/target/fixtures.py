"""Small Ensembl frames shared by the target builder tests."""

from __future__ import annotations

import polars as pl

GENES_SCHEMA = {
    'id': pl.String,
    'biotype': pl.String,
    'description': pl.String,
    'chromosome': pl.String,
    'start': pl.Int64,
    'end': pl.Int64,
    'strand': pl.Int32,
    'approvedSymbol': pl.String,
    'canonicalTranscriptId': pl.String,
}
TRANSCRIPTS_SCHEMA = {
    'geneId': pl.String,
    'transcriptId': pl.String,
    'biotype': pl.String,
    'chromosome': pl.String,
    'start': pl.Int64,
    'end': pl.Int64,
    'strand': pl.Int32,
    'translationId': pl.String,
}
XREFS_SCHEMA = {
    'transcriptId': pl.String,
    'translationId': pl.String,
    'uniprot_swissprot': pl.List(pl.String),
    'uniprot_trembl': pl.List(pl.String),
    'uniprot_isoform': pl.List(pl.String),
    'alphafold': pl.List(pl.String),
    'signalp': pl.List(pl.String),
}

GENES = pl.DataFrame(
    [
        (
            'ENSG00000141510',
            'protein_coding',
            'tumor protein p53 [Source:HGNC Symbol;Acc:HGNC:11998]',
            '17',
            7661779,
            7687546,
            -1,
            'TP53',
            'ENST00000269305',
        ),
        ('ENSG00000000002', 'TEC', 'TEC protein detection [Source:x]', 'X', 1000, 2000, 1, None, 'ENST00000000021'),
        ('ENSG00000000003', 'Mt_tRNA', None, 'MT', 577, 647, 1, 'MT-TF', 'ENST00000000031'),
        ('LRG_1', 'protein_coding', 'lrg', '1', 10, 20, 1, 'LRG', 'LRG_1t1'),
    ],
    schema=GENES_SCHEMA,
    orient='row',
)
TRANSCRIPTS = pl.DataFrame(
    [
        ('ENSG00000000002', 'ENST00000000021', 'TEC', 'X', 1000, 2000, 1, None),
        ('ENSG00000000003', 'ENST00000000031', 'Mt_tRNA', 'MT', 577, 647, 1, None),
        ('ENSG00000141510', 'ENST00000269305', 'protein_coding', '17', 7661779, 7687538, -1, 'ENSP00000269305'),
        ('ENSG00000141510', 'ENST00000413465', 'protein_coding', '17', 7661779, 7687500, -1, 'ENSP00000410739'),
        ('LRG_1', 'LRG_1t1', 'protein_coding', '1', 10, 20, 1, 'LRG_1p1'),
    ],
    schema=TRANSCRIPTS_SCHEMA,
    orient='row',
)
XREFS = pl.DataFrame(
    [
        ('ENST00000269305', 'ENSP00000269305', ['P04637'], ['K7PPA8'], [], [], []),
        ('ENST00000413465', 'ENSP00000410739', [], ['H2EHT1', 'K7PPA8'], [], [], ['SignalP-noTM']),
        ('LRG_1t1', 'LRG_1p1', ['P00001'], [], [], [], []),
    ],
    schema=XREFS_SCHEMA,
    orient='row',
)
