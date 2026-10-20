"""NCBI Entrez gene synonyms."""

from __future__ import annotations

import polars as pl

from pts.transformers.target.lists import flatten_distinct, tag, union
from pts.transformers.utils.dataset import scan_dataset

SYNONYM_COLUMNS = ['synonyms', 'symbolSynonyms', 'nameSynonyms']


def read_ncbi(path: str) -> pl.DataFrame:
    """The NCBI gene_info TSV, every column as text."""
    return (
        scan_dataset(path, format='tsv', has_header=True, quote_char=None, infer_schema=False)
        .select('Symbol', 'dbXrefs', 'Synonyms', 'Other_designations')
        .collect()
    )


def build_ncbi(raw: pl.DataFrame) -> pl.DataFrame:
    """Entrez synonyms per Ensembl gene id named in `dbXrefs`.

    Returns:
        [id, synonyms, symbolSynonyms, nameSynonyms], each a list of {label, source}.
    """
    rows = (
        raw
        .select(
            pl.col('Symbol').str.split('|').alias('sy'),
            pl.col('dbXrefs').str.split('|').alias('id_part'),
            pl.col('Synonyms').str.split('|').alias('s'),
            pl.col('Other_designations').str.split('|').alias('od'),
        )
        .explode('id_part', empty_as_null=False, keep_nulls=False)
        .filter(pl.col('id_part').str.starts_with('Ensembl'))
        .with_columns(pl.col('id_part').str.split(':').alias('id'))
        .explode('id', empty_as_null=False, keep_nulls=False)
        .filter(pl.col('id').str.starts_with('ENSG'))
        .select(
            'id',
            union(pl.col('s'), pl.col('od'), pl.col('sy')).alias('synonyms'),
            union(pl.col('s'), pl.col('sy')).alias('symbolSynonyms'),
            union(pl.col('od')).alias('nameSynonyms'),
        )
    )
    ncbi = rows.select('id').unique(maintain_order=True)
    for column in SYNONYM_COLUMNS:
        ncbi = ncbi.join(flatten_distinct(rows, 'id', column), on='id', how='left', maintain_order='left')
    return ncbi.with_columns([
        tag(pl.col(c).list.eval(pl.element().filter(pl.element() != '-')), 'label', 'NCBI_entrez').alias(c)
        for c in SYNONYM_COLUMNS
    ])
