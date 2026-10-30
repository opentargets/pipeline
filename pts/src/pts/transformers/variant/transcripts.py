"""Per-transcript fields the variant step takes from the Ensembl core tables."""

from __future__ import annotations

import polars as pl

from pts.transformers.utils.ensembl import CoreDump, gene_names, transcripts, translation_xrefs

_MATURE_MIRNA = pl.List(pl.Struct({'start': pl.Int64(), 'end': pl.Int64()}))


def _attribute(code: str) -> pl.Expr:
    return pl.col('attributes').list.eval(
        pl.element().filter(pl.element().struct.field('code') == code).struct.field('value')
    )


def _mature_mirna(frame: pl.DataFrame) -> pl.DataFrame:
    """Genomic intervals of the mature miRNAs, from the cDNA `start-end` of each `miRNA` attribute.

    Returns:
        (`transcript_id`, `mature_mirna`) with one row per transcript that has any. An interval
        spanning an intron is split into one piece per exon.
    """
    exons = (
        frame
        .select('transcript_id', 'strand', 'exons')
        .explode('exons')
        .drop_nulls('exons')
        .select('transcript_id', 'strand', pl.col('exons').struct.field('start', 'end'))
        .with_columns(length=pl.col('end') - pl.col('start') + 1)
        .with_columns(cdna_start=pl.col('length').cum_sum().over('transcript_id') - pl.col('length') + 1)
        .with_columns(cdna_end=pl.col('cdna_start') + pl.col('length') - 1)
    )
    intervals = (
        frame
        .select('transcript_id', _attribute('miRNA').alias('interval'))
        .explode('interval')
        .drop_nulls('interval')
        .with_columns(
            lo=pl.col('interval').str.split('-').list.get(0).cast(pl.Int64()),
            hi=pl.col('interval').str.split('-').list.get(1).cast(pl.Int64()),
        )
    )
    pieces = (
        intervals
        .join(exons, on='transcript_id', how='inner')
        .with_columns(
            first=pl.max_horizontal('lo', 'cdna_start') - pl.col('cdna_start'),
            last=pl.min_horizontal('hi', 'cdna_end') - pl.col('cdna_start'),
        )
        .filter(pl.col('first') <= pl.col('last'))
        .select(
            'transcript_id',
            pl
            .when(pl.col('strand') == 1)
            .then(pl.col('start') + pl.col('first'))
            .otherwise(pl.col('end') - pl.col('last'))
            .alias('start'),
            pl
            .when(pl.col('strand') == 1)
            .then(pl.col('start') + pl.col('last'))
            .otherwise(pl.col('end') - pl.col('first'))
            .alias('end'),
        )
    )
    return (
        pieces
        .sort('transcript_id', 'start', 'end')
        .group_by('transcript_id', maintain_order=True)
        .agg(pl.struct('start', 'end').alias('mature_mirna'))
    )


def transcript_fields(core: str, chromosome: str) -> pl.DataFrame:
    """The transcript fields Ensembl VEP reads from its cache, rebuilt from the core tables.

    Args:
        core: Directory of the Ensembl core database dump.
        chromosome: Chromosome name; only its transcripts are returned.

    Returns:
        One row per transcript, sorted by `transcript_id`. `tss` is the transcript's 5' end,
        `transcript_start` and `transcript_end` are genomic and 1-based, `canonical` is 1 or null,
        `mature_mirna` is `[]` when the transcript has none. Readthrough transcripts are left out.
    """
    dump = CoreDump(core)
    names = gene_names(dump).rename({'id': 'geneId', 'name': 'gene_symbol'})
    xrefs = translation_xrefs(dump).select('transcriptId', 'uniprot_swissprot', 'uniprot_trembl')
    frame = (
        transcripts(dump)
        .filter(pl.col('chromosome') == chromosome)
        .filter(~_attribute('readthrough_tra').list.len().gt(0))
        .rename({'transcriptId': 'transcript_id'})
        .join(names, on='geneId', how='left', maintain_order='left')
        .join(xrefs.rename({'transcriptId': 'transcript_id'}), on='transcript_id', how='left', maintain_order='left')
        .rechunk()
    )
    mirna = _mature_mirna(frame)
    empty = pl.lit([], dtype=pl.List(pl.String()))
    return (
        frame
        .join(mirna, on='transcript_id', how='left', maintain_order='left')
        .select(
            'transcript_id',
            pl.when(_attribute('is_canonical').list.contains('1')).then(pl.lit(1, dtype=pl.Int64())).alias('canonical'),
            _attribute('appris').list.first().alias('appris'),
            _attribute('MANE_Select').list.first().alias('mane_select'),
            'gene_symbol',
            'biotype',
            pl.col('uniprot_swissprot').fill_null(empty).alias('swissprot'),
            pl.col('uniprot_trembl').fill_null(empty).alias('trembl'),
            pl.when(pl.col('strand') == 1).then(pl.col('start')).otherwise(pl.col('end')).alias('tss'),
            pl.col('start').alias('transcript_start'),
            pl.col('end').alias('transcript_end'),
            pl.col('strand').cast(pl.Int8()),
            pl.col('mature_mirna').fill_null(pl.lit([], dtype=_MATURE_MIRNA)),
        )
        .sort('transcript_id')
    )


def canonical_transcript_ids(core: str) -> set[str]:
    """The stable ids of the Ensembl canonical transcripts on every chromosome, from the core tables."""
    frame = transcripts(CoreDump(core))
    return set(frame.filter(_attribute('is_canonical').list.contains('1'))['transcriptId'].to_list())
