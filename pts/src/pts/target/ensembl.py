"""Ensembl genes: coordinates, names, protein ids and the canonical transcript."""

from __future__ import annotations

import polars as pl

from pts.ensembl import CANONICAL_CHROMOSOMES, CoreDump, genes, transcripts, translation_xrefs
from pts.target.lists import tag, union

TEC = '[Tt][Ee][Cc]'


def build_ensembl(dump: CoreDump) -> pl.DataFrame:
    """Ensembl genes from the core database dump."""
    return ensembl_genes(genes(dump), transcripts(dump), translation_xrefs(dump))


def ensembl_genes(genes: pl.DataFrame, transcripts: pl.DataFrame, translation_xrefs: pl.DataFrame) -> pl.DataFrame:
    """One row per gene.

    Args:
        genes: `pts.ensembl.genes`.
        transcripts: `pts.ensembl.transcripts`.
        translation_xrefs: `pts.ensembl.translation_xrefs`.

    Returns:
        [id, biotype, approvedName, genomicLocation, approvedSymbol, proteinIds, signalP,
        canonicalTranscript].
    """
    proteins = (
        transcripts
        .select('geneId', 'translationId')
        .join(
            translation_xrefs.select('translationId', 'uniprot_swissprot', 'uniprot_trembl', 'signalp'),
            on='translationId',
            how='left',
        )
        .group_by('geneId')
        .agg(
            pl.col('translationId').drop_nulls().unique().sort().alias('ensembl_PRO'),
            *[
                pl.col(c).explode(empty_as_null=False, keep_nulls=False).drop_nulls().unique().sort().alias(c)
                for c in ('uniprot_swissprot', 'uniprot_trembl', 'signalp')
            ],
        )
    )
    canonical = transcripts.select(
        pl.col('geneId').alias('id'),
        pl.col('transcriptId').alias('canonicalTranscriptId'),
        pl.struct(
            pl.col('transcriptId').alias('id'),
            'chromosome',
            'start',
            'end',
            'strand',
        ).alias('canonicalTranscript'),
    )
    return (
        genes
        .join(proteins, left_on='id', right_on='geneId', how='left')
        .with_columns(pl.col('ensembl_PRO', 'uniprot_swissprot', 'uniprot_trembl', 'signalp').fill_null([]))
        .filter(pl.col('id').str.starts_with('ENSG'))
        .filter(pl.col('chromosome').is_in(CANONICAL_CHROMOSOMES))
        .with_columns(pl.col('id').str.strip_chars(' '))
        .sort('id', maintain_order=True)
        .unique(subset='id', keep='first', maintain_order=True)
        .join(canonical, on=['id', 'canonicalTranscriptId'], how='left', maintain_order='left')
        .rechunk()
        .select(
            'id',
            pl.col('biotype').str.replace_all(TEC, ''),
            pl.col('description').str.split('[').list.first().str.replace_all(TEC, '').alias('approvedName'),
            pl.struct('chromosome', 'start', 'end', 'strand').alias('genomicLocation'),
            'approvedSymbol',
            union(
                tag(pl.col('uniprot_swissprot'), 'id', 'uniprot_swissprot'),
                tag(pl.col('uniprot_trembl'), 'id', 'uniprot_trembl'),
                tag(pl.col('ensembl_PRO'), 'id', 'ensembl_PRO'),
            ).alias('proteinIds'),
            tag(pl.col('signalp'), 'id', 'signalP').alias('signalP'),
            pl
            .when(pl.col('canonicalTranscript').struct.field('chromosome') == pl.col('chromosome'))
            .then(pl.col('canonicalTranscript'))
            .alias('canonicalTranscript'),
        )
    )
