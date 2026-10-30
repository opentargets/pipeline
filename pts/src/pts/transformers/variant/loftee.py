"""LOFTEE's HC/LC call and filters on canonical transcript consequences, as the VEP plugin makes them.

The plugin runs with the production parameters (`konradjk/loftee` branch `grch38`, `a46b502`):
every option at its default, so splice predictions, `INCOMPLETE_CDS` and the PhyloCSF flags are off.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from pts.schemas.ensembl_core import EXON, TRANSCRIPT
from pts.transformers.utils.ensembl import CoreDump, transcripts
from pts.transformers.variant.alleles import vep_allele_form
from pts.transformers.variant.gerp import KentBigWig

LOF_TERMS = ['stop_gained', 'frameshift_variant', 'splice_acceptor_variant', 'splice_donor_variant']
MIN_INTRON_SIZE = 15
GERP_END_TRUNC_CUTOFF = -58

_TRANSLATION = {
    'transcript_id': pl.Int64(),
    'seq_start': pl.Int64(),
    'start_exon_id': pl.Int64(),
    'seq_end': pl.Int64(),
    'end_exon_id': pl.Int64(),
}
_COMPLEMENT = str.maketrans('ACGTNacgtn', 'TGCANtgcan')
_CALLS = pl.Schema({'lof': pl.String, 'lof_filter': pl.String})


@dataclass(frozen=True)
class TranscriptModel:
    """A transcript as LOFTEE reads it from Ensembl.

    `exons` are (start, end) in transcript order, genomic and 1-based. `coding_start` and
    `coding_end` are the genomic bounds of the CDS, stop codon included; None without a translation.
    """

    strand: int
    biotype: str
    exons: tuple[tuple[int, int], ...]
    coding_start: int | None
    coding_end: int | None

    @property
    def introns(self) -> list[tuple[int, int]]:
        """(start, end) of the gap between each pair of consecutive exons, in transcript order."""
        pairs = zip(self.exons, self.exons[1:], strict=False)
        if self.strand == 1:
            return [(previous[1] + 1, current[0] - 1) for previous, current in pairs]
        return [(current[1] + 1, previous[0] - 1) for previous, current in pairs]


def _coding_regions(dump: CoreDump) -> pl.DataFrame:
    """The genomic CDS bounds of each translated transcript, from its translation's start and end exons."""
    transcript = dump.scan('transcript', TRANSCRIPT).select(
        'transcript_id', pl.col('stable_id').alias('transcriptId'), pl.col('seq_region_strand').alias('strand')
    )
    exon = dump.scan('exon', EXON)
    first = exon.select(
        pl.col('exon_id').alias('start_exon_id'),
        pl.col('seq_region_start').alias('first_start'),
        pl.col('seq_region_end').alias('first_end'),
    )
    last = exon.select(
        pl.col('exon_id').alias('end_exon_id'),
        pl.col('seq_region_start').alias('last_start'),
        pl.col('seq_region_end').alias('last_end'),
    )
    plus = pl.col('strand') == 1
    begin = (
        pl
        .when(plus)
        .then(pl.col('first_start') + pl.col('seq_start') - 1)
        .otherwise(pl.col('first_end') - pl.col('seq_start') + 1)
    )
    end = (
        pl
        .when(plus)
        .then(pl.col('last_start') + pl.col('seq_end') - 1)
        .otherwise(pl.col('last_end') - pl.col('seq_end') + 1)
    )
    return (
        dump
        .scan('translation', _TRANSLATION)
        .join(transcript, on='transcript_id', how='inner')
        .join(first, on='start_exon_id', how='inner')
        .join(last, on='end_exon_id', how='inner')
        .select(
            'transcriptId',
            pl.min_horizontal(begin, end).alias('coding_start'),
            pl.max_horizontal(begin, end).alias('coding_end'),
        )
        .collect()
    )


def loftee_models(core: str, chromosome: str) -> dict[str, TranscriptModel]:
    """The transcripts of one chromosome as LOFTEE reads them, from the Ensembl core tables.

    Args:
        core: Directory of the Ensembl core database dump.
        chromosome: Chromosome name; only its transcripts are returned.

    Returns:
        Transcript stable id to its model.
    """
    dump = CoreDump(core)
    frame = (
        transcripts(dump)
        .filter(pl.col('chromosome') == chromosome)
        .join(_coding_regions(dump), on='transcriptId', how='left', maintain_order='left')
    )
    return {
        row['transcriptId']: TranscriptModel(
            strand=row['strand'],
            biotype=row['biotype'],
            exons=tuple((exon['start'], exon['end']) for exon in row['exons']),
            coding_start=row['coding_start'],
            coding_end=row['coding_end'],
        )
        for row in frame.iter_rows(named=True)
    }


def _reverse_complement(sequence: str) -> str:
    return sequence.translate(_COMPLEMENT)[::-1]


def _number(features: list[tuple[int, int]] | tuple[tuple[int, int], ...], start: int, end: int) -> int | None:
    """The 1-based number of the first feature the variant overlaps, as Perl reads `N-M/T`."""
    for number, (low, high) in enumerate(features, start=1):
        if low <= high and end >= low and start <= high:
            return number
    return None


def _gerp_weighted_dist(model: TranscriptModel, chromosome: str, position: int, gerp: KentBigWig) -> tuple[float, int]:
    """LOFTEE `get_gerp_weighted_dist`: GERP summed, and bases counted, from the variant to the stop codon.

    Every exon not upstream of the variant contributes, including exons past the stop codon.
    """
    plus = model.strand == 1
    stop = (model.coding_end if plus else model.coding_start) or 0
    weighted, distance = 0.0, 0
    for low, high in model.exons:
        if (plus and position > high) or (not plus and position < low):
            continue
        last = (low < stop <= high) if plus else (low <= stop < high)
        affected = low <= position <= high
        if last:
            start = position if affected else (low if plus else high)
            end = stop
        elif affected:
            start = position
            end = high if plus else low
        else:
            start, end = low, high
        weighted = weighted + gerp.interval_gerp(chromosome, start, end)
        distance = distance + abs(end - start)
    return weighted, distance


def _last_exon_coding_length(model: TranscriptModel, exon_index: int) -> int:
    """LOFTEE `get_last_exon_coding_length`, searching from the last exon back to `exon_index`."""
    plus = model.strand == 1
    stop = (model.coding_end if plus else model.coding_start) or 0
    for low, high in reversed(model.exons[exon_index:]):
        if plus:
            if low > stop:
                continue
            if high >= stop:
                return stop - low
        else:
            if high < stop:
                continue
            if low <= stop:
                return high - stop
    return -1000


def _filters(
    model: TranscriptModel,
    terms: list[str],
    start: int,
    end: int,
    alleles: tuple[str, str],
    cds_end: int | None,
    sequence: bytes,
    gerp: KentBigWig,
    chromosome: str,
) -> list[str]:
    """The filters LOFTEE pushes before `ANC_ALLELE`, in the plugin's order."""
    filters = []
    plus = model.strand == 1
    coding_start, coding_end = model.coding_start or 0, model.coding_end or 0
    other_lof = 'stop_gained' in terms or 'frameshift_variant' in terms
    vep_splice = 'splice_acceptor_variant' in terms or 'splice_donor_variant' in terms
    exon = _number(model.exons, start, end)
    introns = model.introns
    intron = _number(introns, start, end)
    if other_lof and cds_end is not None:
        weighted, distance = _gerp_weighted_dist(model, chromosome, start, gerp)
        if exon:
            last_exon = distance - _last_exon_coding_length(model, exon - 1)
            if last_exon <= 50 and weighted <= GERP_END_TRUNC_CUTOFF:
                filters.append('END_TRUNC')
    if intron:
        low, high = introns[intron - 1]
        if high - low + 1 < MIN_INTRON_SIZE:
            filters.append('SMALL_INTRON')
        if vep_splice:
            if 'splice_donor_variant' in terms:
                if plus:
                    motif = sequence[low - 1 : low + 1].decode()
                    reference, alternate = alleles
                else:
                    motif = _reverse_complement(sequence[high - 2 : high].decode())
                    reference, alternate = (_reverse_complement(allele) for allele in alleles)
                if motif == 'GC' and reference == 'C' and alternate == 'T':
                    filters.append('GC_TO_GT_DONOR')
            if (end < coding_start) if plus else (start > coding_end):
                filters.append('5UTR_SPLICE')
            if (start > coding_end) if plus else (end < coding_start):
                filters.append('3UTR_SPLICE')
    return filters


def _ancestral_bases(ancestor: bytes, positions: set[int]) -> dict[int, str]:
    """The ancestral base at each position of the contig; none beyond its end."""
    return {position: chr(ancestor[position - 1]) for position in positions if 1 <= position <= len(ancestor)}


def loftee(
    records: pl.DataFrame,
    models: dict[str, TranscriptModel],
    sequence: bytes,
    ancestor: bytes,
    gerp: KentBigWig,
    chromosome: str,
) -> pl.DataFrame:
    """LOFTEE's call and filters for each transcript consequence.

    A consequence with a LoF term on a `protein_coding` transcript is `HC`, or `LC` when any
    filter fires: `END_TRUNC`, `SMALL_INTRON`, `GC_TO_GT_DONOR`, `5UTR_SPLICE`, `3UTR_SPLICE`
    and `ANC_ALLELE`, joined with `,` in that order. GERP is read through kent's zoom levels.
    A chromosome the bigWig or the ancestor lacks scores no GERP or has no ancestral allele.

    Args:
        records: One row per transcript consequence, with VEP's `vepStart`, `vepEnd`,
            `vepReference` and `vepAlternate`, and `transcript_id`, `consequence_terms` and `cds_end`.
        models: `loftee_models` of the chromosome.
        sequence: The chromosome's reference, byte i being position i + 1.
        ancestor: The chromosome's GRCh38 human ancestor in upper case, byte i being position
            i + 1; empty for a chromosome the ancestor lacks.
        gerp: The GERP bigWig.
        chromosome: The chromosome of the records.

    Returns:
        `lof` and `lof_filter`, row for row; both null without a call.
    """
    has_lof_term = pl.col('consequence_terms').list.eval(pl.element().is_in(LOF_TERMS)).list.any().fill_null(False)
    candidates = (
        records
        .with_row_index('_index')
        .filter(has_lof_term)
        .select(
            '_index',
            'transcript_id',
            'consequence_terms',
            'vepStart',
            'vepEnd',
            'vepReference',
            'vepAlternate',
            'cds_end',
        )
    )
    called: dict[int, list[str]] = {}
    substitutions: dict[int, tuple[int, str]] = {}
    for row, transcript_id, terms, start, end, reference, alternate, cds_end in candidates.iter_rows():
        model = models.get(transcript_id)
        if model is None or model.biotype != 'protein_coding':
            continue
        called[row] = _filters(model, terms, start, end, (reference, alternate), cds_end, sequence, gerp, chromosome)
        if '-' not in (reference, alternate) and len(reference) == len(alternate) == 1:
            substitutions[row] = (start, alternate.upper())
    ancestral = _ancestral_bases(ancestor, {start for start, _ in substitutions.values()})
    lof: list[str | None] = [None] * records.height
    lof_filter: list[str | None] = [None] * records.height
    for row, filters in called.items():
        if row in substitutions:
            start, alternate = substitutions[row]
            if ancestral.get(start, '') == alternate:
                filters.append('ANC_ALLELE')
        lof[row] = 'LC' if filters else 'HC'
        lof_filter[row] = ','.join(filters) if filters else None
    return pl.DataFrame({'lof': lof, 'lof_filter': lof_filter}, schema=_CALLS)


def add_loftee(
    records: pl.LazyFrame,
    models: dict[str, TranscriptModel],
    sequence: bytes,
    ancestor: bytes,
    gerp: KentBigWig,
    chromosome: str,
) -> pl.LazyFrame:
    """Fill `lof` and `lof_filter` of the canonical transcript consequences of the records.

    Coordinates and alleles are VEP's, from each record's `input` line; consequence terms and
    `cds_end` are the consequence's own. See `loftee`.

    Args:
        records: The VEP-shaped records of one chromosome, with VCF-form `input` lines.
        models: `loftee_models` of the chromosome.
        sequence: The chromosome's reference, byte i being position i + 1.
        ancestor: The chromosome's GRCh38 human ancestor in upper case, byte i being position
            i + 1; empty for a chromosome the ancestor lacks.
        gerp: The GERP bigWig.
        chromosome: The chromosome of the records.

    Returns:
        The records, row for row; other transcript consequences are unchanged.
    """
    frame = records.collect()
    fields = pl.col('input').str.split('\t')
    variants = vep_allele_form(
        frame.select(
            fields.list.get(1).cast(pl.Int64).alias('position'),
            fields.list.get(3).alias('referenceAllele'),
            fields.list.get(4).alias('alternateAllele'),
        ).lazy()
    ).collect()
    consequences = (
        frame
        .select(pl.col('transcript_consequences').alias('transcript'))
        .with_columns(
            pl.Series('row', range(frame.height), dtype=pl.UInt32),
            *variants.select('vepStart', 'vepEnd', 'vepReference', 'vepAlternate').get_columns(),
        )
        .filter(pl.col('transcript').list.len() > 0)
        .explode('transcript')
        .rechunk()
        .with_columns(pl.col('transcript').struct.field('canonical', 'transcript_id', 'consequence_terms', 'cds_end'))
    )
    canonical = pl.col('canonical') == 1
    calls = loftee(
        consequences.with_columns(pl.when(canonical).then('consequence_terms').alias('consequence_terms')),
        models,
        sequence,
        ancestor,
        gerp,
        chromosome,
    )
    filled = (
        consequences
        .with_columns(calls.select(pl.col('lof').alias('_lof'), pl.col('lof_filter').alias('_lof_filter')))
        .rechunk()
        .select(
            'row',
            pl.col('transcript').struct.with_fields(
                pl.when(canonical).then('_lof').otherwise(pl.field('lof')).alias('lof'),
                pl.when(canonical).then('_lof_filter').otherwise(pl.field('lof_filter')).alias('lof_filter'),
            ),
        )
        .group_by('row', maintain_order=True)
        .agg(pl.col('transcript').alias('filled'))
    )
    return (
        frame
        .with_row_index('row')
        .join(filled, on='row', how='left', maintain_order='left')
        .with_columns(pl.coalesce('filled', 'transcript_consequences').alias('transcript_consequences'))
        .drop('row', 'filled')
        .select(frame.columns)
        .lazy()
    )
