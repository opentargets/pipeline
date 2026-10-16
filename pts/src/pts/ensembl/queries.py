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


def _canonical_transcripts(dump: CoreDump) -> pl.LazyFrame:
    """Transcripts of the genes on a canonical chromosome, with the gene's stable id as `geneId`."""
    gene = _canonical_genes(dump).select('gene_id', pl.col('stable_id').alias('geneId'))
    return dump.scan('transcript', schema.TRANSCRIPT).join(gene, on='gene_id', how='inner')


def _attributes(dump: CoreDump) -> pl.LazyFrame:
    """Per transcript, the kept attributes sorted by (code, value)."""
    codes = dump.scan('attrib_type', schema.ATTRIB_TYPE).filter(pl.col('code').is_in(TRANSCRIPT_ATTRIBUTES))
    return (
        dump
        .scan('transcript_attrib', schema.TRANSCRIPT_ATTRIB)
        .join(codes, on='attrib_type_id', how='inner')
        .sort('transcript_id', 'code', 'value')
        .group_by('transcript_id', maintain_order=True)
        .agg(pl.struct('code', 'value').alias('attributes'))
    )


def _exons(dump: CoreDump) -> pl.LazyFrame:
    """Per transcript, its exons ordered by `exon_transcript.rank`."""
    return (
        dump
        .scan('exon_transcript', schema.EXON_TRANSCRIPT)
        .join(dump.scan('exon', schema.EXON), on='exon_id', how='inner')
        .join(_seq_region_names(dump), on='seq_region_id', how='left')
        .sort('transcript_id', 'rank')
        .group_by('transcript_id', maintain_order=True)
        .agg(
            pl.struct(
                pl.col('stable_id').alias('exonId'),
                'chromosome',
                pl.col('seq_region_start').alias('start'),
                pl.col('seq_region_end').alias('end'),
                pl.col('seq_region_strand').alias('strand'),
            ).alias('exons')
        )
    )


def transcripts(dump: CoreDump) -> pl.DataFrame:
    """Transcripts of the genes on the canonical chromosomes.

    Returns:
        One row per transcript, sorted by (`geneId`, `transcriptId`), with the columns of
        `schemas.ensembl_core.transcripts`. `translationId` is null for a non-coding transcript;
        `attributes` and `exons` are `[]` when there are none.
    """
    translation = dump.scan('translation', schema.TRANSLATION).select(
        'transcript_id', pl.col('stable_id').alias('translationId')
    )
    return (
        _canonical_transcripts(dump)
        .join(_seq_region_names(dump), on='seq_region_id', how='left')
        .join(translation, on='transcript_id', how='left')
        .join(_attributes(dump), on='transcript_id', how='left')
        .join(_exons(dump), on='transcript_id', how='left')
        .select(
            'geneId',
            pl.col('stable_id').alias('transcriptId'),
            'biotype',
            'chromosome',
            pl.col('seq_region_start').alias('start'),
            pl.col('seq_region_end').alias('end'),
            pl.col('seq_region_strand').alias('strand'),
            'translationId',
            pl.col('attributes').fill_null(pl.lit([], dtype=pl.List(schema.attribute))),
            pl.col('exons').fill_null(pl.lit([], dtype=pl.List(schema.exon))),
        )
        .sort('geneId', 'transcriptId')
        .collect()
    )


def _xref_values(dump: CoreDump) -> pl.LazyFrame:
    """(translation_id, field, value) for every accession and protein feature reported."""
    external_db = dump.scan('external_db', schema.EXTERNAL_DB).filter(pl.col('db_name').is_in(list(_UNIPROT_DBS)))
    accessions = (
        dump
        .scan('object_xref', schema.OBJECT_XREF)
        .filter(pl.col('ensembl_object_type') == 'Translation')
        .join(dump.scan('xref', schema.XREF), on='xref_id', how='inner')
        .join(external_db, on='external_db_id', how='inner')
        .select(
            pl.col('ensembl_id').alias('translation_id'),
            pl.col('db_name').replace_strict(_UNIPROT_DBS, return_dtype=pl.String()).alias('field'),
            pl.col('dbprimary_acc').alias('value'),
        )
    )
    analysis = dump.scan('analysis', schema.ANALYSIS).filter(pl.col('logic_name').is_in(_FEATURE_ANALYSES))
    features = (
        dump
        .scan('protein_feature', schema.PROTEIN_FEATURE)
        .join(analysis, on='analysis_id', how='inner')
        .select('translation_id', pl.col('logic_name').alias('field'), pl.col('hit_name').alias('value'))
    )
    return pl.concat([accessions, features])


def translation_xrefs(dump: CoreDump) -> pl.DataFrame:
    """UniProt accessions and protein features of the translations of `transcripts`.

    LRG transcripts share translation stable ids, so the chromosome filter is also what keeps
    `translationId` unique.

    Returns:
        One row per translation, sorted by `translationId`, with the columns of
        `schemas.ensembl_core.translation_xrefs`. Each list is unique and sorted, `[]` when empty.
    """
    per_translation = (
        _xref_values(dump)
        .unique()
        .sort('translation_id', 'field', 'value')
        .group_by('translation_id', maintain_order=True)
        .agg([pl.col('value').filter(pl.col('field') == field).alias(field) for field in _XREF_FIELDS])
    )
    transcript = _canonical_transcripts(dump).select('transcript_id', pl.col('stable_id').alias('transcriptId'))
    empty = pl.lit([], dtype=pl.List(pl.String()))
    return (
        dump
        .scan('translation', schema.TRANSLATION)
        .join(transcript, on='transcript_id', how='inner')
        .join(per_translation, on='translation_id', how='left')
        .select(
            'transcriptId',
            pl.col('stable_id').alias('translationId'),
            *[pl.col(field).fill_null(empty) for field in _XREF_FIELDS],
        )
        .sort('translationId')
        .collect()
    )
