"""COSMIC cancer hallmarks by gene symbol."""

from __future__ import annotations

import polars as pl

from pts.transformers.utils.dataset import scan_dataset
from pts.transformers.utils.spark_casts import to_long

CANCER_HALLMARKS = [
    'proliferative signalling',
    'invasion and metastasis',
    'suppression of growth',
    'angiogenesis',
    'change of cellular energetics',
    'genome instability and mutations',
    'escaping programmed cell death',
    'tumour promoting inflammation',
    'cell replicative immortality',
    'escaping immune response to cancer',
]


COLUMNS = ['GENE_SYMBOL', 'PUBMED_PMID', 'HALLMARK', 'IMPACT', 'DESCRIPTION']


def unquote(col: pl.Expr) -> pl.Expr:
    """Strip the quotes around a field, as spark's CSV reader does.

    A quoted field holding another quote is not a quoted value to spark, which keeps it verbatim.
    """
    inner = col.str.slice(1, col.str.len_chars() - 2)
    quoted = col.str.starts_with('"') & col.str.ends_with('"') & (col.str.len_chars() >= 2)
    return pl.when(quoted & ~inner.str.contains('"', literal=True)).then(inner).otherwise(col)


def read_hallmarks(path: str) -> pl.DataFrame:
    """The hallmarks TSV columns used here, as text."""
    return (
        scan_dataset(path, format='tsv', has_header=True, quote_char=None, infer_schema=False)
        .select([unquote(pl.col(c)).alias(c) for c in COLUMNS])
        .collect()
    )


def _collect(frame: pl.DataFrame, fields: dict[str, str], name: str) -> pl.DataFrame:
    """Distinct `fields` structs per gene symbol, sorted."""
    return (
        frame
        .select('gene_symbol', *[pl.col(src).alias(dst) for dst, src in fields.items()])
        .unique()
        .sort(pl.all(), nulls_last=False)
        .group_by('gene_symbol', maintain_order=True)
        .agg(pl.struct(*fields).alias(name))
    )


def build_hallmarks(raw: pl.DataFrame) -> pl.DataFrame:
    """Hallmarks per gene symbol.

    Returns:
        [approvedSymbol, hallmarks{attributes, cancerHallmarks}].
    """
    processed = raw.select(
        pl.col('GENE_SYMBOL').alias('gene_symbol'),
        to_long(pl.col('PUBMED_PMID')).alias('pmid'),
        pl.col('HALLMARK').alias('hallmark'),
        pl.col('IMPACT').alias('impact'),
        pl.col('DESCRIPTION').alias('description'),
    ).with_columns(pl.col('hallmark').is_in(CANCER_HALLMARKS).alias('is_cancer'))

    cancer = _collect(
        processed.filter(pl.col('is_cancer')),
        {'pmid': 'pmid', 'description': 'description', 'impact': 'impact', 'label': 'hallmark'},
        'cancerHallmarks',
    )
    attributes = _collect(
        processed.filter(~pl.col('is_cancer')),
        {'pmid': 'pmid', 'description': 'description', 'name': 'hallmark'},
        'attributes',
    )
    return (
        processed
        .select('gene_symbol')
        .unique(maintain_order=True)
        .join(cancer, on='gene_symbol', how='left', maintain_order='left')
        .join(attributes, on='gene_symbol', how='left', maintain_order='left')
        .select(
            pl.col('gene_symbol').alias('approvedSymbol'),
            pl.struct('attributes', 'cancerHallmarks').alias('hallmarks'),
        )
    )
