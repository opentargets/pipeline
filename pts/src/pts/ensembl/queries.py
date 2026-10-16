"""Shared views over an Ensembl core dump.

Each query returns raw values from the dump as an eager frame with an explicit sort order. Mapping
them onto published fields (strand spelling, null versus empty list, flag names) is the caller's job.
"""

from __future__ import annotations

import polars as pl

from pts.ensembl.core import CoreDump
from pts.schemas import ensembl_core as schema

CANONICAL_CHROMOSOMES = [str(i) for i in range(1, 23)] + ['X', 'Y', 'MT']

_CHROMOSOME_COORD_SYSTEM = 'chromosome'

#: `external_db.db_name` to the output column its accessions go to.
_UNIPROT_DBS = {
    'Uniprot/SWISSPROT': 'uniprot_swissprot',
    'Uniprot/SPTREMBL': 'uniprot_trembl',
    'Uniprot_isoform': 'uniprot_isoform',
}

#: `analysis.logic_name` of the protein features whose `hit_name` is reported.
_FEATURE_ANALYSES = ['alphafold', 'signalp']

_XREF_FIELDS = [*_UNIPROT_DBS.values(), *_FEATURE_ANALYSES]

#: `attrib_type.code` values kept on a transcript.
TRANSCRIPT_ATTRIBUTES = ['is_canonical', 'MANE_Select', 'MANE_Plus_Clinical', 'gencode_primary', 'appris']


def _seq_region_names(dump: CoreDump) -> pl.LazyFrame:
    """Every seq_region's name, keyed by `seq_region_id`."""
    return dump.scan('seq_region', schema.SEQ_REGION).select('seq_region_id', pl.col('name').alias('chromosome'))


def _canonical_seq_regions(dump: CoreDump) -> pl.LazyFrame:
    """The seq_regions that are canonical chromosomes."""
    coord_systems = (
        dump
        .scan('coord_system', schema.COORD_SYSTEM)
        .filter(pl.col('name') == _CHROMOSOME_COORD_SYSTEM)
        .select('coord_system_id')
    )
    return (
        dump
        .scan('seq_region', schema.SEQ_REGION)
        .join(coord_systems, on='coord_system_id', how='semi')
        .filter(pl.col('name').is_in(CANONICAL_CHROMOSOMES))
        .select('seq_region_id', pl.col('name').alias('chromosome'))
    )


def _canonical_genes(dump: CoreDump) -> pl.LazyFrame:
    """Genes on a canonical chromosome, with their chromosome name."""
    return dump.scan('gene', schema.GENE).join(_canonical_seq_regions(dump), on='seq_region_id', how='inner')


def gene_names(dump: CoreDump) -> pl.DataFrame:
    """Every gene's stable id and display name, on any seq_region.

    Returns:
        One row per gene, sorted by `id`; `name` is null when the gene has no display xref.
    """
    xref = dump.scan('xref', schema.XREF).select('xref_id', 'display_label')
    return (
        dump
        .scan('gene', schema.GENE)
        .join(xref, left_on='display_xref_id', right_on='xref_id', how='left')
        .select(pl.col('stable_id').alias('id'), pl.col('display_label').alias('name'))
        .sort('id')
        .collect()
    )


def genes(dump: CoreDump) -> pl.DataFrame:
    """Genes on the canonical chromosomes.

    Returns:
        One row per gene, sorted by `id`, with the columns of `schemas.ensembl_core.genes`.
    """
    xref = dump.scan('xref', schema.XREF).select('xref_id', 'display_label')
    transcript = dump.scan('transcript', schema.TRANSCRIPT).select(
        pl.col('transcript_id').alias('canonical_transcript_id'),
        pl.col('stable_id').alias('canonicalTranscriptId'),
    )
    return (
        _canonical_genes(dump)
        .join(xref, left_on='display_xref_id', right_on='xref_id', how='left')
        .join(transcript, on='canonical_transcript_id', how='left')
        .select(
            pl.col('stable_id').alias('id'),
            'biotype',
            'description',
            'chromosome',
            pl.col('seq_region_start').alias('start'),
            pl.col('seq_region_end').alias('end'),
            pl.col('seq_region_strand').alias('strand'),
            pl.col('display_label').alias('approvedSymbol'),
            'canonicalTranscriptId',
        )
        .sort('id')
        .collect()
    )
