"""HGNC symbols, names, synonyms and UniProt mappings."""

from __future__ import annotations

import json

import polars as pl
from otter.storage.synchronous.handle import StorageHandle

from pts.schemas.target import HGNC_DOC
from pts.transformers.target.lists import tag


def read_hgnc(path: str) -> pl.DataFrame:
    """The `response.docs` of the HGNC complete set JSON, in file order."""
    data, _ = StorageHandle(path).read()
    docs = json.loads(data)['response']['docs']
    return pl.DataFrame([{k: doc.get(k) for k in HGNC_DOC} for doc in docs], schema=HGNC_DOC)


def build_hgnc(docs: pl.DataFrame) -> pl.DataFrame:
    """One row per Ensembl gene id, the first HGNC doc naming it.

    Returns:
        [ensemblId, hgncId, approvedSymbol, approvedName, uniprotIds, hgncSymbolSynonyms,
        hgncNameSynonyms, hgncObsoleteSymbols, hgncObsoleteNames].
    """
    parts = pl.col('hgnc_id').str.split(':')
    return docs.select(
        pl.col('ensembl_gene_id').alias('ensemblId'),
        pl.concat_list(
            pl.struct(
                parts.list.get(1, null_on_oob=True).alias('id'),
                parts.list.get(0, null_on_oob=True).alias('source'),
            )
        ).alias('hgncId'),
        pl.col('symbol').alias('approvedSymbol'),
        pl.col('name').alias('approvedName'),
        pl.col('uniprot_ids').alias('uniprotIds'),
        tag(pl.col('alias_symbol').fill_null([]), 'label', 'HGNC').alias('hgncSymbolSynonyms'),
        tag(pl.col('alias_name').fill_null([]), 'label', 'HGNC').alias('hgncNameSynonyms'),
        tag(pl.col('prev_symbol').fill_null([]), 'label', 'HGNC').alias('hgncObsoleteSymbols'),
        tag(pl.col('prev_name').fill_null([]), 'label', 'HGNC').alias('hgncObsoleteNames'),
    ).unique(subset='ensemblId', keep='first', maintain_order=True)
