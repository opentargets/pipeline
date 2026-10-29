"""Target lookups shared by transformers."""

from __future__ import annotations

import polars as pl

#: The `proteinIds` sources ChEMBL accessions are resolved through.
UNIPROT_SOURCES = ['uniprot_swissprot', 'uniprot_trembl']


def ensg_lookup(target: pl.LazyFrame) -> pl.LazyFrame:
    """Map each protein accession and approved symbol of a target to its ENSG id.

    Args:
        target: the target dataset.

    Returns:
        One row per distinct (`ensgId`, `name`) pair.
    """
    proteins = target.select(
        pl.col('id').alias('ensgId'),
        pl.col('proteinIds').list.eval(pl.element().struct.field('id')).alias('name'),
    ).explode('name')
    symbols = target.select(pl.col('id').alias('ensgId'), pl.col('approvedSymbol').alias('name'))
    return pl.concat([proteins, symbols]).drop_nulls('name').unique(maintain_order=True)


def tss_lut(target: pl.LazyFrame) -> pl.LazyFrame:
    """Resolve gene TSS from the preferred `tss`, then strand-aware genomic coordinates.

    Gentropy also fell back to the canonical transcript, but the PTS target index has no
    `canonicalTranscript` column, so that step is not ported.
    """
    genomic = pl.col('genomicLocation')
    genomic_tss = (
        pl
        .when(genomic.struct.field('strand') == 1)
        .then(genomic.struct.field('start'))
        .when(genomic.struct.field('strand') == -1)
        .then(genomic.struct.field('end'))
    )
    return target.select(
        pl.col('id').alias('geneId'),
        pl.coalesce(pl.col('tss'), genomic_tss).cast(pl.Int64).alias('tss'),
    )


def chembl_accession_lookup(gene_df: pl.DataFrame) -> pl.DataFrame:
    """Map each accession a ChEMBL target component can carry to its Ensembl gene id.

    ChEMBL spells a target as a uniprot accession or an Ensembl gene id, so resolution
    needs both lookups. `genes_by_id` covers the whole of `gene_df`, not just the rows
    surviving the explode, which drops genes with no accession.

    Args:
        gene_df: The target dataset's `id` and `proteinIds`.

    Returns:
        Distinct (`geneId`, `uniprot_id`) pairs, in no particular order.
    """
    genes_by_uniprot = (
        gene_df
        .select(pl.col('id').alias('geneId'), pl.col('proteinIds').alias('protein'))
        .explode('protein', empty_as_null=False, keep_nulls=False)
        .filter(pl.col('protein').struct.field('source').is_in(UNIPROT_SOURCES))
        .select('geneId', pl.col('protein').struct.field('id').alias('uniprot_id'))
    )
    genes_by_id = gene_df.select(pl.col('id').alias('geneId'), pl.col('id').alias('uniprot_id'))
    return pl.concat([genes_by_uniprot, genes_by_id]).unique()
