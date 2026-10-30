"""The fastVEP JSON output read into the VEP records that `parse_vep` reads."""

from __future__ import annotations

import io
import mmap
from pathlib import Path
from typing import cast

import numpy as np
import polars as pl

from pts.schemas.vep import vep_schema
from pts.transformers.variant.alleles import hgvs_genomic, vep_allele_form
from pts.transformers.variant.consequences import RANKS

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
_TRANSCRIPT = cast(pl.Struct, cast(pl.List, vep_schema['transcript_consequences']).inner)
_BASES = r'^[ACGTN]+$'


def read_records(path: Path, chunk_bytes: int = CHUNK_BYTES) -> pl.DataFrame:
    """The fastVEP JSON array, parsed in slices of about `chunk_bytes` cut at record boundaries.

    Returns:
        One row per record: `row`, the row ordinal fastVEP echoes as the record's `id`,
        `most_severe_consequence` and every `transcript_consequences` entry.
    """
    frames = [pl.DataFrame(schema=FASTVEP_SCHEMA)]
    with path.open('rb') as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as buffer:
        start = buffer.find(_RECORD) + 1
        while 0 < start < len(buffer):
            cut = buffer.find(_RECORD, start + chunk_bytes)
            end = len(buffer) if cut == -1 else cut + 1
            piece = buffer[start:end].rstrip().rstrip(b']').rstrip().rstrip(b',')
            frames.append(pl.read_json(io.BytesIO(b'[' + piece + b']'), schema=FASTVEP_SCHEMA))
            start = end
    return pl.concat(frames).select(
        pl.col('id').cast(pl.Int64).alias('row'), 'most_severe_consequence', 'transcript_consequences'
    )


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
    """The rows with their ordinal, VEP's `input` line, VEP's coordinates and the genomic HGVS."""
    return (
        vep_allele_form(rows.lazy().with_row_index('row').with_columns(pl.col('row').cast(pl.Int64)))
        .collect()
        .with_columns(
            pl.concat_str(
                'chromosome',
                pl.col('position').cast(pl.String),
                pl.col('rsId').fill_null('.'),
                'referenceAllele',
                'alternateAllele',
                pl.lit('.\t.\t.'),
                separator='\t',
            ).alias('input'),
            _hgvsg(rows, sequence, chromosome),
        )
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
            variants.select('row', 'vepStart', 'vepEnd', 'vepReference', 'vepAlternate', 'hgvsg'),
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
    halves = pl.col('amino_acids').str.split_exact('/', 1)
    synonymous = halves.struct.field('field_0') == halves.struct.field('field_1')
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
        pl
        .when(~ablation)
        .then(pl.when(synonymous).then(halves.struct.field('field_0')).otherwise('amino_acids'))
        .alias('amino_acids'),
        pl.when(~ablation).then('codons').alias('codons'),
        pl.when(~ablation).then('protein_end').alias('protein_end'),
    )


def _transcript() -> pl.Expr:
    """The VEP transcript consequence struct; fields other tasks fill are null."""
    fields = {
        'amino_acids': pl.col('amino_acids'),
        'hgvsg': pl.col('hgvsg'),
        'gene_symbol': pl.col('gene_symbol'),
        'biotype': pl.col('biotype'),
        'appris': pl
        .col('appris')
        .str.replace('principal', 'P', literal=True)
        .str.replace('alternative', 'A', literal=True),
        'mane_select': pl.col('mane_select'),
        'tssdistance': pl
        .when(pl.col('strand') == 1)
        .then(pl.col('tss') - pl.col('vepEnd'))
        .otherwise(pl.col('vepStart') - pl.col('tss')),
        'canonical': pl.col('core_canonical'),
        'cds_end': pl.col('cds_end'),
        'codons': pl.col('codons'),
        'consequence_terms': pl.col('consequence_terms'),
        'distance': pl.col('distance'),
        'gene_id': pl.col('gene_id'),
        'impact': pl.col('impact'),
        'protein_end': pl.col('protein_end'),
        'swissprot': pl.when(pl.col('swissprot').list.len() > 0).then(pl.col('swissprot')),
        'transcript_id': pl.col('transcript_id'),
        'trembl': pl.when(pl.col('trembl').list.len() > 0).then(pl.col('trembl')),
    }
    return pl.struct([
        fields[name].cast(dtype).alias(name) if name in fields else pl.lit(None, dtype).alias(name)
        for name, dtype in _TRANSCRIPT.to_schema().items()
    ])


def fastvep_records(
    output: Path, rows: pl.DataFrame, transcripts: pl.DataFrame, sequence: bytes, chromosome: str
) -> pl.LazyFrame:
    """The fastVEP output of one chromosome as the VEP records `parse_vep` reads.

    Records join `rows` on the ordinal fastVEP echoes as their `id`. `input` is rebuilt from
    the original row and `hgvsg` from the reference. Transcript fields come from `transcripts`
    (`transcript_fields`), with APPRIS written as VEP does (`P1`, `A2`). VEP's
    `transcript_ablation` and `mature_miRNA_variant` are applied, and the most severe
    consequence is ranked over every transcript fastVEP reported, by VEP's ranks. Only
    transcripts canonical in `transcripts` are kept. A row without a transcript gets `intergenic_consequences`.
    `colocated_variants` is empty; `conservation`, `alphamissense`, `lof` and `lof_filter` are null.

    Raises:
        ValueError: if the record ids are not the row ordinals of `rows`.
    """
    records = read_records(output)
    if records.height != rows.height:
        raise ValueError(f'{output}: {records.height} fastVEP records for {rows.height} rows')
    if records['row'].sort().to_list() != list(range(rows.height)):
        raise ValueError(f'{output}: fastVEP record ids are not the ordinals 0..{rows.height - 1} of the rows')
    variants = _variants(rows, sequence, chromosome)
    consequences = _consequences(records, variants, transcripts)
    rank = pl.col('consequence_terms').list.eval(
        pl.element().replace_strict(_RANK, default=len(RANKS), return_dtype=pl.Int32)
    )
    severity = consequences.group_by('row').agg(
        rank.list.min().min().replace_strict(_TERM, default=None, return_dtype=pl.String).alias('_most_severe'),
        pl.len().alias('_transcripts'),
    )
    canonical = (
        consequences
        .filter(pl.col('core_canonical') == 1)
        .group_by('row', maintain_order=True)
        .agg(_transcript().alias('transcript_consequences'))
    )
    located = vep_schema['colocated_variants']
    return (
        variants
        .select('row', 'input', 'hgvsg')
        .join(records.select('row', 'most_severe_consequence'), on='row', how='left', maintain_order='left')
        .join(severity, on='row', how='left', maintain_order='left')
        .join(canonical, on='row', how='left', maintain_order='left')
        .rechunk()
        .select(
            'input',
            pl.coalesce('_most_severe', 'most_severe_consequence').alias('most_severe_consequence'),
            pl
            .when(pl.col('_transcripts').is_null())
            .then(pl.concat_list(pl.struct('hgvsg')))
            .cast(vep_schema['intergenic_consequences'])
            .alias('intergenic_consequences'),
            pl.lit([], dtype=located).alias('colocated_variants'),
            pl.col('transcript_consequences').fill_null(pl.lit([], dtype=vep_schema['transcript_consequences'])),
        )
        .lazy()
    )
