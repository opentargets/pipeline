"""The fastVEP JSON output read into the variants and consequences frames of a chunk."""

from __future__ import annotations

import io
import mmap
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import polars as pl

from pts.transformers.variant.alleles import hgvs_genomic, vep_allele_form
from pts.transformers.variant.consequences import RANKS, consequence_score
from pts.transformers.variant.frames import (
    CONSEQUENCES_SCHEMA,
    COPIED,
    FIELDS,
    VARIANT_COLUMNS,
    ChunkFrames,
)

FASTVEP_SCHEMA = pl.Schema({
    'id': pl.String,
    'most_severe_consequence': pl.String,
    'transcript_consequences': pl.List(
        pl.Struct({
            'transcript_id': pl.String,
            'gene_id': pl.String,
            'gene_symbol': pl.String,
            'biotype': pl.String,
            'canonical': pl.Int64,
            'cds_end': pl.Int64,
            'consequence_terms': pl.List(pl.String),
            'distance': pl.Int64,
            'impact': pl.String,
            'amino_acids': pl.String,
            'codons': pl.String,
            'protein_end': pl.Int64,
        })
    ),
})
"""The fastVEP JSON fields the records are built from; the rest of each record is skipped."""

CHUNK_BYTES = 256 << 20
_RANK = {term: rank for rank, term in enumerate(RANKS)}
_TERM = dict(enumerate(RANKS))
_RECORD = b'\n{\n'
_BASES = r'^[ACGTN]+$'


def iter_records(path: Path, chunk_bytes: int = CHUNK_BYTES) -> Iterator[pl.DataFrame]:
    """The fastVEP JSON array, parsed in slices of about `chunk_bytes` cut at record boundaries.

    Yields:
        One frame per slice, one row per record: `row`, the row ordinal fastVEP echoes as the
        record's `id`, `most_severe_consequence` and every `transcript_consequences` entry.
    """
    with path.open('rb') as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as buffer:
        start = buffer.find(_RECORD) + 1
        while 0 < start < len(buffer):
            cut = buffer.find(_RECORD, start + chunk_bytes)
            end = len(buffer) if cut == -1 else cut + 1
            piece = buffer[start:end].rstrip().rstrip(b']').rstrip().rstrip(b',')
            yield pl.read_json(io.BytesIO(b'[' + piece + b']'), schema=FASTVEP_SCHEMA).select(
                pl.col('id').cast(pl.Int64).alias('row'), 'most_severe_consequence', 'transcript_consequences'
            )
            start = end


def _hgvsg(rows: pl.DataFrame, sequence: bytes, chromosome: str) -> pl.Series:
    """`hgvs_genomic` of each row; null for alleles other than bases or beyond the sequence."""
    ref, alt = pl.col('referenceAllele'), pl.col('alternateAllele')
    frame = rows.select(
        pl.col('position').cast(pl.Int64),
        ref,
        alt,
        (
            ref.str.contains(_BASES)
            & alt.str.contains(_BASES)
            & (pl.col('position') >= 1)
            & (pl.col('position') + ref.str.len_chars() - 1 <= len(sequence))
        ).alias('valid'),
        ((ref.str.len_chars() == 1) & (alt.str.len_chars() == 1)).alias('snv'),
    )
    values: list[str | None] = [None] * frame.height
    kind = 'm' if chromosome == 'MT' else 'g'
    snv = frame.with_row_index('i').filter(pl.col('valid') & pl.col('snv'))
    bases = np.frombuffer(sequence, dtype=np.uint8)[snv['position'].to_numpy() - 1].tobytes().decode()
    for i, position, base, allele in zip(snv['i'], snv['position'], bases, snv['alternateAllele'], strict=True):
        if base != allele:
            values[i] = f'{chromosome}:{kind}.{position}{base}>{allele}'
    other = frame.with_row_index('i').filter(pl.col('valid') & ~pl.col('snv') & (ref != alt))
    for i, position, reference, alternate in other.select('i', 'position', ref, alt).iter_rows():
        values[i] = hgvs_genomic(sequence, chromosome, position, reference, alternate)
    return pl.Series('hgvsg', values, dtype=pl.String)


def _variants(rows: pl.DataFrame, sequence: bytes, chromosome: str) -> pl.DataFrame:
    """The rows with their ordinal, VEP's coordinates and the genomic HGVS."""
    return (
        vep_allele_form(rows.lazy().with_row_index('row').with_columns(pl.col('row').cast(pl.Int64)))
        .collect()
        .with_columns(_hgvsg(rows, sequence, chromosome))
    )


def _mirna_hits(consequences: pl.DataFrame) -> pl.DataFrame:
    """(`row`, `transcript_id`) pairs whose VEP span overlaps a mature miRNA of the transcript."""
    return (
        consequences
        .select('row', 'transcript_id', 'vepStart', 'vepEnd', 'mature_mirna')
        .filter(pl.col('mature_mirna').list.len() > 0)
        .explode('mature_mirna')
        .filter(
            (pl.col('mature_mirna').struct.field('start') <= pl.col('vepEnd'))
            & (pl.col('mature_mirna').struct.field('end') >= pl.col('vepStart'))
        )
        .select('row', 'transcript_id', pl.lit(True).alias('mirna'))
        .unique(['row', 'transcript_id'])
    )


def _consequences(records: pl.DataFrame, variants: pl.DataFrame, transcripts: pl.DataFrame) -> pl.DataFrame:
    """The transcript consequences fastVEP reported, one row each, with the transcript fields and VEP's rules."""
    joined = (
        records
        .select('row', 'transcript_consequences')
        .explode('transcript_consequences')
        .unnest('transcript_consequences')
        .filter(pl.col('transcript_id') != '-')
        .join(
            transcripts.drop('gene_symbol', 'biotype').rename({'canonical': 'core_canonical'}),
            on='transcript_id',
            how='left',
            maintain_order='left',
        )
        .join(
            variants.select('row', 'vepStart', 'vepEnd', 'vepReference', 'vepAlternate'),
            on='row',
            how='left',
            maintain_order='left',
        )
    )
    joined = joined.join(_mirna_hits(joined), on=['row', 'transcript_id'], how='left', maintain_order='left').rechunk()
    length = pl.when(pl.col('vepAlternate') == '-').then(0).otherwise(pl.col('vepAlternate').str.len_chars())
    deletion = length < pl.when(pl.col('vepReference') == '-').then(0).otherwise(pl.col('vepReference').str.len_chars())
    ablation = (
        deletion & (pl.col('vepStart') <= pl.col('transcript_start')) & (pl.col('vepEnd') >= pl.col('transcript_end'))
    ).fill_null(False)
    terms = pl.col('consequence_terms')
    return joined.with_columns(
        pl
        .when(ablation)
        .then(pl.concat_list(pl.lit('transcript_ablation')))
        .when(pl.col('mirna'))
        .then(
            terms.list.eval(
                pl
                .when(pl.element() == 'non_coding_transcript_exon_variant')
                .then(pl.lit('mature_miRNA_variant'))
                .otherwise(pl.element())
            )
        )
        .otherwise(terms)
        .alias('consequence_terms'),
        pl.when(ablation).then(pl.lit('HIGH')).otherwise('impact').alias('impact'),
        pl.when(~ablation).then('amino_acids').alias('amino_acids'),
        pl.when(~ablation).then('codons').alias('codons'),
        pl.when(~ablation).then('protein_end').alias('protein_end'),
    )


def _fields() -> list[pl.Expr]:
    """The transcript fields of the consequences frame, written as VEP writes them."""
    computed = {
        'appris': pl
        .col('appris')
        .str.replace('principal', 'P', literal=True)
        .str.replace('alternative', 'A', literal=True),
        'tssdistance': pl
        .when(pl.col('strand') == 1)
        .then(pl.col('tss') - pl.col('vepEnd'))
        .otherwise(pl.col('vepStart') - pl.col('tss')),
        'swissprot': pl.when(pl.col('swissprot').list.len() > 0).then(pl.col('swissprot')),
        'trembl': pl.when(pl.col('trembl').list.len() > 0).then(pl.col('trembl')),
    }
    return [computed.get(name, pl.col(name)).cast(dtype).alias(name) for name, dtype in FIELDS.items()]


def rank_consequences(consequences: pl.DataFrame) -> pl.DataFrame:
    """The consequences with `rank`: highest consequence score, then distance, then transcript id, within a row."""
    return (
        consequences
        .with_row_index('_i')
        .with_columns(
            consequence_score(pl.col('consequence_terms')).alias('_score'),
            pl.col('distance').fill_null(0).alias('_distance'),
        )
        .sort(
            ['row', '_score', '_distance', 'transcript_id'],
            descending=[False, True, False, False],
            nulls_last=True,
            maintain_order=True,
        )
        .with_columns(pl.int_range(1, pl.len() + 1, dtype=pl.Int32).over('row').alias('rank'))
        .sort('_i')
        .drop('_i', '_score', '_distance')
    )


def chunk_frames(
    records: pl.DataFrame, rows: pl.DataFrame, transcripts: pl.DataFrame, sequence: bytes, chromosome: str
) -> ChunkFrames:
    """The records `iter_records` reads for rows of one chromosome, as a variants and a consequences frame.

    Records join `rows` on the ordinal fastVEP echoes as their `id`, the row's ordinal in `rows`.
    `hgvsg` is rebuilt from the reference. Transcript fields come from `transcripts`
    (`transcript_fields`), with APPRIS written as VEP does (`P1`, `A2`). VEP's `transcript_ablation`
    and `mature_miRNA_variant` are applied, and the most severe consequence is ranked over every
    transcript fastVEP reported, by VEP's ranks. `consequences` keeps only transcripts canonical in
    `transcripts`, in fastVEP's order, each with its `rank` among them.
    """
    variants = _variants(rows, sequence, chromosome)
    reported = _consequences(records, variants, transcripts)
    rank = pl.col('consequence_terms').list.eval(
        pl.element().replace_strict(_RANK, default=len(RANKS), return_dtype=pl.Int32)
    )
    severity = reported.group_by('row').agg(
        rank.list.min().min().replace_strict(_TERM, default=None, return_dtype=pl.String).alias('_most_severe'),
        pl.len().alias('_transcripts'),
    )
    canonical = (
        reported
        .filter(pl.col('core_canonical') == 1)
        .join(
            variants.select('row', 'position', 'referenceAllele', 'alternateAllele'),
            on='row',
            how='left',
            maintain_order='left',
        )
        .select(*COPIED, *_fields())
    )
    variants = (
        variants
        .join(records.select('row', 'most_severe_consequence'), on='row', how='left', maintain_order='left')
        .join(severity, on='row', how='left', maintain_order='left')
        .rechunk()
        .select(
            *VARIANT_COLUMNS,
            pl.coalesce('_most_severe', 'most_severe_consequence').alias('most_severe_consequence'),
            pl.col('_transcripts').is_not_null().alias('reported_transcripts'),
        )
    )
    return ChunkFrames(variants, rank_consequences(canonical))


def unannotated_frames(rows: pl.DataFrame, chromosome: str) -> ChunkFrames:
    """The frames of rows on a chromosome without reference sequence.

    Each variant is intergenic, with a null `hgvsg` and no transcript consequence.
    """
    variants = _variants(rows, b'', chromosome).select(
        *VARIANT_COLUMNS,
        pl.lit('intergenic_variant').alias('most_severe_consequence'),
        pl.lit(False).alias('reported_transcripts'),
    )
    return ChunkFrames(variants, pl.DataFrame(schema=CONSEQUENCES_SCHEMA))
