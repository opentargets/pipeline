"""Tests for LOFTEE's HC/LC call and filters on the variant step's transcript consequences."""

from __future__ import annotations

import json
import struct
import zlib
from collections.abc import Iterator
from pathlib import Path

import polars as pl
import pyBigWig  # ty: ignore[unresolved-import] -- compiled module without stubs
import pytest

from pts.transformers.variant.alleles import read_chromosome
from pts.transformers.variant.frames import CONSEQUENCES_SCHEMA
from pts.transformers.variant.gerp import KentBigWig
from pts.transformers.variant.loftee import (
    GERP_END_TRUNC_CUTOFF,
    MIN_INTRON_SIZE,
    TranscriptModel,
    add_loftee,
    loftee,
    loftee_models,
)
from pts.transformers.variant.prepared import _sequence
from tests.ensembl_fixtures import cells, ddl, write_dump
from tests.variant_fixtures import consequences

ATM = json.loads((Path(__file__).parent / 'test_data' / 'loftee_atm_ENST00000675843.json').read_text())
LENGTH = 4000
FASTA_WIDTH = 100
BLOCK_BASES = 600 * FASTA_WIDTH

#: Hand-built transcripts on contig 1. GERP scores -2 on bases 1 to 1000 and +2 on 1001 to 4000.
MODELS = {
    'TP': TranscriptModel(1, 'protein_coding', ((101, 200), (301, 400), (501, 700)), 151, 650),
    'TM': TranscriptModel(-1, 'protein_coding', ((900, 950), (801, 850)), 810, 940),
    'TQ': TranscriptModel(1, 'protein_coding', ((1101, 1200), (1301, 1400), (1501, 1700)), 1151, 1650),
    'TU': TranscriptModel(1, 'protein_coding', ((2101, 2200), (2301, 2400), (2501, 2600)), 2320, 2390),
    'TS': TranscriptModel(1, 'protein_coding', ((2701, 2750), (2761, 2800)), 2710, 2790),
    'TI': TranscriptModel(1, 'protein_coding', ((3101, 3200), (3301, 3400)), 3150, 3352),
    'TN': TranscriptModel(1, 'nonsense_mediated_decay', ((3501, 3600),), 3510, 3590),
}

#: Bases other than `A` in the reference: intron motifs, and the reference of the GC donor variants.
MOTIFS = {
    201: 'GT', 299: 'AG',  # TP intron 1
    401: 'GC', 499: 'AG',  # TP intron 2, a GC donor
    851: 'CT', 898: 'GC',  # TM intron, GC on the minus strand
    1201: 'AT', 1299: 'AC',  # TQ intron 1, a non-canonical motif
    1401: 'GT', 1499: 'AG',
    2201: 'GT', 2299: 'AG', 2401: 'GT', 2499: 'AG',
    2751: 'GT', 2759: 'AG',  # TS intron, 10 bases
}  # fmt: skip

#: Where the ancestor differs from the reference.
ANCESTRAL = {160: 'T', 600: 'T'}


def genome() -> bytes:
    bases = bytearray(b'A' * LENGTH)
    for start, motif in MOTIFS.items():
        bases[start - 1 : start - 1 + len(motif)] = motif.encode()
    return bytes(bases)


def bgzf_block(data: bytes) -> bytes:
    """One BGZF block: a gzip member with the BC extra field giving its size."""
    compressor = zlib.compressobj(9, zlib.DEFLATED, -15)
    deflated = compressor.compress(data) + compressor.flush()
    header = struct.pack('<BBBBIBBHBBHH', 31, 139, 8, 4, 0, 0, 255, 6, 66, 67, 2, 18 + len(deflated) + 8 - 1)
    return header + deflated + struct.pack('<II', zlib.crc32(data), len(data))


def _chunks(length: int, start: int, window: str) -> Iterator[tuple[bool, bytes]]:
    """FASTA lines of a contig in blocks of BLOCK_BASES bases: N everywhere but `window` at `start`."""
    end = start + len(window) - 1
    for first in range(1, length + 1, BLOCK_BASES):
        last = min(first + BLOCK_BASES - 1, length)
        plain = last < start or first > end
        if plain and last - first + 1 == BLOCK_BASES:
            yield True, b''
            continue
        bases = bytearray(b'N' * (last - first + 1))
        for position in range(max(first, start), min(last, end) + 1):
            bases[position - first] = ord(window[position - start])
        lines = [bases[i : i + FASTA_WIDTH] for i in range(0, len(bases), FASTA_WIDTH)]
        yield False, b''.join(bytes(line) + b'\n' for line in lines)


def write_bgzip_fasta(path: Path, contigs: dict[str, tuple[int, int, str]]) -> str:
    """A bgzip FASTA, from contig name to (length, start, window)."""
    repeated = bgzf_block(b''.join(b'N' * FASTA_WIDTH + b'\n' for _ in range(BLOCK_BASES // FASTA_WIDTH)))
    with path.open('wb') as out:
        for name, (length, start, window) in contigs.items():
            out.write(bgzf_block(f'>{name}\n'.encode()))
            for is_full, data in _chunks(length, start, window):
                out.write(repeated if is_full else bgzf_block(data))
        out.write(bgzf_block(b''))
    return str(path)


@pytest.fixture(scope='module')
def ancestor_fasta(tmp_path_factory: pytest.TempPathFactory) -> str:
    """The reference with the ANCESTRAL bases, lower case, on contig 1 only."""
    bases = bytearray(genome())
    for position, base in ANCESTRAL.items():
        bases[position - 1] = ord(base)
    sequence = bases.decode().lower()
    path = tmp_path_factory.mktemp('ancestor') / 'human_ancestor.fa.gz'
    return write_bgzip_fasta(path, {'1': (LENGTH, 1, sequence)})


@pytest.fixture(scope='module')
def ancestor(ancestor_fasta: str) -> bytes:
    return read_chromosome(ancestor_fasta, '1')


@pytest.fixture(scope='module')
def gerp(tmp_path_factory: pytest.TempPathFactory) -> Iterator[KentBigWig]:
    path = str(tmp_path_factory.mktemp('gerp') / 'gerp.bw')
    handle = pyBigWig.open(path, 'w')
    handle.addHeader([('1', LENGTH)])
    handle.addEntries(['1', '1'], [0, 1000], ends=[1000, LENGTH], values=[-2.0, 2.0])
    handle.close()
    with KentBigWig(path) as bigwig:
        yield bigwig


def consequence(
    transcript_id: str,
    terms: list[str],
    start: int,
    reference: str,
    alternate: str,
    *,
    end: int | None = None,
    cds_end: int | None = 10,
) -> dict:
    """A transcript consequence in VEP's coordinates and alleles."""
    return {
        'vepStart': start,
        'vepEnd': start + len(reference) - 1 if end is None else end,
        'vepReference': reference,
        'vepAlternate': alternate,
        'transcript_id': transcript_id,
        'consequence_terms': terms,
        'cds_end': cds_end,
    }


CONSEQUENCES = pl.Schema({
    'vepStart': pl.Int64,
    'vepEnd': pl.Int64,
    'vepReference': pl.String,
    'vepAlternate': pl.String,
    'transcript_id': pl.String,
    'consequence_terms': pl.List(pl.String),
    'cds_end': pl.Int64,
})


def calls(
    ancestor: bytes,
    gerp: KentBigWig,
    *rows: dict,
    chromosome: str = '1',
    sequence: bytes | None = None,
    models: dict[str, TranscriptModel] | None = None,
) -> list[tuple[str | None, str | None]]:
    records = pl.DataFrame(list(rows), schema=CONSEQUENCES)
    result = loftee(
        records,
        MODELS if models is None else models,
        genome() if sequence is None else sequence,
        ancestor,
        gerp,
        chromosome,
    )
    assert result.schema == pl.Schema({'lof': pl.String, 'lof_filter': pl.String})
    return list(result.iter_rows())


def test_defaults():
    assert MIN_INTRON_SIZE == 15
    assert GERP_END_TRUNC_CUTOFF == -58


def test_stop_gained_far_from_the_end_is_high_confidence(ancestor: bytes, gerp: KentBigWig):
    assert calls(ancestor, gerp, consequence('TP', ['stop_gained'], 170, 'A', 'T')) == [('HC', None)]


def test_end_trunc(ancestor: bytes, gerp: KentBigWig):
    # 40 bases from the stop in the last exon, where GERP is -2 a base: -82 <= -58
    assert calls(ancestor, gerp, consequence('TP', ['stop_gained'], 610, 'A', 'T')) == [('LC', 'END_TRUNC')]


def test_end_trunc_needs_low_gerp(ancestor: bytes, gerp: KentBigWig):
    # the same position on TQ, where GERP is +2 a base
    assert calls(ancestor, gerp, consequence('TQ', ['stop_gained'], 1610, 'A', 'T')) == [('HC', None)]


def test_end_trunc_needs_a_cds_end(ancestor: bytes, gerp: KentBigWig):
    assert calls(ancestor, gerp, consequence('TP', ['stop_gained'], 610, 'A', 'T', cds_end=None)) == [('HC', None)]


def test_end_trunc_on_an_insertion_inside_the_last_exon(ancestor: bytes, gerp: KentBigWig):
    # an insertion between 620 and 621 ends before it starts; 30 bases at -2 give -60
    row = consequence('TP', ['frameshift_variant'], 621, '-', 'T', end=620)
    assert calls(ancestor, gerp, row) == [('LC', 'END_TRUNC')]


def test_end_trunc_on_a_deletion(ancestor: bytes, gerp: KentBigWig):
    assert calls(ancestor, gerp, consequence('TP', ['frameshift_variant'], 611, 'A', '-')) == [('LC', 'END_TRUNC')]


def test_incomplete_cds_is_not_checked(ancestor: bytes, gerp: KentBigWig):
    # TI's CDS is 103 bases, not whole codons; check_complete_cds is off in production
    assert calls(ancestor, gerp, consequence('TI', ['stop_gained'], 3160, 'A', 'T')) == [('HC', None)]


def test_exon_intron_undef_never_fires(ancestor: bytes, gerp: KentBigWig):
    # a stop_gained on no exon of TP: LOFTEE only tests for a missing exon number when it has one
    assert calls(ancestor, gerp, consequence('TP', ['stop_gained'], 250, 'A', 'T')) == [('HC', None)]


def test_small_intron(ancestor: bytes, gerp: KentBigWig):
    assert calls(ancestor, gerp, consequence('TS', ['splice_donor_variant'], 2751, 'G', 'A')) == [
        ('LC', 'SMALL_INTRON')
    ]


def test_non_canonical_splice_is_only_a_flag(ancestor: bytes, gerp: KentBigWig):
    assert calls(ancestor, gerp, consequence('TQ', ['splice_donor_variant'], 1201, 'A', 'G')) == [('HC', None)]


def test_gc_to_gt_donor(ancestor: bytes, gerp: KentBigWig):
    assert calls(
        ancestor,
        gerp,
        consequence('TP', ['splice_donor_variant'], 402, 'C', 'T'),
        consequence('TP', ['splice_donor_variant'], 402, 'C', 'A'),
        # on the minus strand the genome's G at 898 is the intron's C
        consequence('TM', ['splice_donor_variant'], 898, 'G', 'A'),
    ) == [('LC', 'GC_TO_GT_DONOR'), ('HC', None), ('LC', 'GC_TO_GT_DONOR')]


def test_utr_splice(ancestor: bytes, gerp: KentBigWig):
    assert calls(
        ancestor,
        gerp,
        consequence('TU', ['splice_donor_variant'], 2201, 'G', 'A'),
        consequence('TU', ['splice_acceptor_variant'], 2500, 'G', 'A'),
    ) == [('LC', '5UTR_SPLICE'), ('LC', '3UTR_SPLICE')]


def test_anc_allele(ancestor: bytes, gerp: KentBigWig):
    assert calls(ancestor, gerp, consequence('TP', ['stop_gained'], 160, 'A', 'T')) == [('LC', 'ANC_ALLELE')]


def test_filters_in_plugin_order(ancestor: bytes, gerp: KentBigWig):
    assert calls(ancestor, gerp, consequence('TP', ['stop_gained'], 600, 'A', 'T')) == [('LC', 'END_TRUNC,ANC_ALLELE')]


def test_anc_allele_only_for_single_base_substitutions(ancestor: bytes, gerp: KentBigWig):
    # the ancestor has T at 160, but an insertion or an untrimmed pair is never checked
    assert calls(
        ancestor,
        gerp,
        consequence('TP', ['frameshift_variant'], 161, '-', 'T', end=160),
        consequence('TP', ['stop_gained'], 160, 'AA', 'TT'),
    ) == [('HC', None), ('HC', None)]


def test_no_call(ancestor: bytes, gerp: KentBigWig):
    assert (
        calls(
            ancestor,
            gerp,
            consequence('TN', ['stop_gained'], 3550, 'A', 'T'),
            consequence('UNKNOWN', ['stop_gained'], 160, 'A', 'T'),
            consequence('TP', ['missense_variant'], 160, 'A', 'T'),
        )
        == [(None, None)] * 3
    )


def test_chromosome_missing_from_ancestor_and_gerp(gerp: KentBigWig):
    # MT is in neither file: no GERP and no ancestral allele, so neither END_TRUNC nor ANC_ALLELE
    assert calls(
        b'',
        gerp,
        consequence('TP', ['stop_gained'], 600, 'A', 'T'),
        consequence('TP', ['splice_donor_variant'], 402, 'C', 'T'),
        chromosome='MT',
        sequence=b'',
    ) == [('HC', None), ('HC', None)]


def test_contig_missing_from_the_ancestor_has_no_ancestral_allele(ancestor_fasta: str, gerp: KentBigWig):
    """The ancestor has contig 1 only: read for MT it is empty, and the ANC_ALLELE call at 160 does not fire."""
    assert calls(read_chromosome(ancestor_fasta, '1'), gerp, consequence('TP', ['stop_gained'], 160, 'A', 'T')) == [
        ('LC', 'ANC_ALLELE')
    ]
    missing = _sequence(ancestor_fasta, 'MT')
    assert missing == b''
    assert calls(missing, gerp, consequence('TP', ['stop_gained'], 160, 'A', 'T')) == [('HC', None)]


def test_order_and_empty_input(ancestor: bytes, gerp: KentBigWig):
    assert calls(
        ancestor,
        gerp,
        consequence('TP', ['missense_variant'], 170, 'A', 'T'),
        consequence('TP', ['stop_gained'], 610, 'A', 'T'),
        consequence('TP', ['stop_gained'], 170, 'A', 'T'),
    ) == [(None, None), ('LC', 'END_TRUNC'), ('HC', None)]
    assert calls(ancestor, gerp) == []


def test_ancestor_is_grch38(tmp_path: Path):
    """`11_108289761_C_T`, a stop_gained on ATM's canonical transcript, is HC: the GRCh38 ancestor has C there.

    The window around it, the transcript and the ancestor slice are real; see the fixture's provenance.
    """
    start = ATM['window_start']
    position = 108_289_761
    assert ATM['reference'][position - start] == 'C'
    assert ATM['ancestor'][position - start].upper() == 'C'
    ancestor = read_chromosome(
        write_bgzip_fasta(tmp_path / 'human_ancestor.fa.gz', {'11': (135_086_622, start, ATM['ancestor'])}), '11'
    )
    sequence = bytearray(b'N' * (start + len(ATM['reference'])))
    sequence[start - 1 :] = ATM['reference'].encode()
    bigwig = str(tmp_path / 'gerp.bw')
    handle = pyBigWig.open(bigwig, 'w')
    handle.addHeader([('11', 135_086_622)])
    handle.addEntries(['11'], [0], ends=[10], values=[0.0])
    handle.close()
    model = TranscriptModel(
        ATM['strand'],
        ATM['biotype'],
        tuple(tuple(exon) for exon in ATM['exons']),
        ATM['coding_start'],
        ATM['coding_end'],
    )
    row = consequence(ATM['transcript_id'], ['stop_gained'], position, 'C', 'T', cds_end=ATM['cds_end'])
    with KentBigWig(bigwig) as gerp:
        result = calls(
            ancestor, gerp, row, chromosome='11', sequence=bytes(sequence), models={ATM['transcript_id']: model}
        )
        # the same call against an ancestor that has the alternate allele there
        derived = ATM['ancestor'][: position - start] + 'T' + ATM['ancestor'][position - start + 1 :]
        other = read_chromosome(
            write_bgzip_fasta(tmp_path / 'derived.fa.gz', {'11': (135_086_622, start, derived)}), '11'
        )
        flipped = calls(
            other, gerp, row, chromosome='11', sequence=bytes(sequence), models={ATM['transcript_id']: model}
        )
    assert result == [('HC', None)]
    assert flipped == [('LC', 'ANC_ALLELE')]


def test_add_loftee_adds_the_calls_in_row_order(ancestor: bytes, gerp: KentBigWig):
    rows = [
        consequence('TP', ['stop_gained'], 610, 'A', 'T'),
        consequence('TQ', ['missense_variant'], 610, 'A', 'T'),
        consequence('TP', ['frameshift_variant'], 170, 'GA', 'G'),
    ]
    frame = consequences(row=[0, 0, 1], **{name: [row[name] for row in rows] for name in CONSEQUENCES.names()})
    filled = add_loftee(frame, MODELS, genome(), ancestor, gerp, '1')
    assert filled.columns == [*CONSEQUENCES_SCHEMA.names(), 'lof', 'lof_filter']
    assert filled.drop('lof', 'lof_filter').equals(frame)
    assert filled.select('transcript_id', 'lof', 'lof_filter').rows() == [
        ('TP', 'LC', 'END_TRUNC'),
        ('TQ', None, None),
        ('TP', 'HC', None),
    ]


def test_effects_accept_an_empty_consequences_frame(ancestor: bytes, gerp: KentBigWig):
    found = add_loftee(pl.DataFrame(schema=CONSEQUENCES_SCHEMA), MODELS, genome(), ancestor, gerp, '1')
    assert found.height == 0
    assert found.select('lof', 'lof_filter').schema == pl.Schema({'lof': pl.String, 'lof_filter': pl.String})


def test_loftee_models_from_the_core(tmp_path: Path):
    tables: dict[str, tuple[list[str], list[tuple[object, ...]]]] = {
        'coord_system': (['coord_system_id', 'species_id', 'name'], [(1, 1, 'chromosome')]),
        'seq_region': (
            ['seq_region_id', 'name', 'coord_system_id', 'length'],
            [(10, '1', 1, 5000), (11, '2', 1, 5000)],
        ),
        'gene': (
            [
                'gene_id', 'biotype', 'seq_region_id', 'seq_region_start', 'seq_region_end', 'seq_region_strand',
                'display_xref_id', 'description', 'canonical_transcript_id', 'stable_id',
            ],
            [
                (1, 'protein_coding', 10, 101, 700, 1, None, None, 100, 'G_PLUS'),
                (2, 'protein_coding', 10, 801, 950, -1, None, None, 200, 'G_MINUS'),
                (3, 'lncRNA', 10, 1001, 1100, 1, None, None, 300, 'G_NONCODING'),
                (4, 'protein_coding', 11, 1, 100, 1, None, None, 400, 'G_OTHER'),
            ],
        ),
        'transcript': (
            [
                'transcript_id', 'gene_id', 'seq_region_id', 'seq_region_start', 'seq_region_end',
                'seq_region_strand', 'biotype', 'canonical_translation_id', 'stable_id',
            ],
            [
                (100, 1, 10, 101, 700, 1, 'protein_coding', 1000, 'T_PLUS'),
                (200, 2, 10, 801, 950, -1, 'protein_coding', 2000, 'T_MINUS'),
                (300, 3, 10, 1001, 1100, 1, 'lncRNA', None, 'T_NONCODING'),
                (400, 4, 11, 1, 100, 1, 'protein_coding', 4000, 'T_OTHER'),
            ],
        ),
        'translation': (
            ['translation_id', 'transcript_id', 'seq_start', 'start_exon_id', 'seq_end', 'end_exon_id', 'stable_id'],
            [
                # plus: 51st base of exon 1 to the 150th of exon 3
                (1000, 100, 51, 1, 150, 3, 'P_PLUS'),
                # minus: 11th base of exon 4 (from its 3' end, 950) to the 40th of exon 5 (from 850)
                (2000, 200, 11, 4, 40, 5, 'P_MINUS'),
                (4000, 400, 1, 7, 99, 7, 'P_OTHER'),
            ],
        ),
        'exon': (
            ['exon_id', 'seq_region_id', 'seq_region_start', 'seq_region_end', 'seq_region_strand', 'stable_id'],
            [
                (1, 10, 101, 200, 1, 'E1'),
                (2, 10, 301, 400, 1, 'E2'),
                (3, 10, 501, 700, 1, 'E3'),
                (4, 10, 900, 950, -1, 'E4'),
                (5, 10, 801, 850, -1, 'E5'),
                (6, 10, 1001, 1100, 1, 'E6'),
                (7, 11, 1, 100, 1, 'E7'),
            ],
        ),
        # ranks out of row order
        'exon_transcript': (
            ['exon_id', 'transcript_id', 'rank'],
            [(3, 100, 3), (1, 100, 1), (2, 100, 2), (5, 200, 2), (4, 200, 1), (6, 300, 1), (7, 400, 1)],
        ),
        'attrib_type': (['attrib_type_id', 'code', 'name'], [(554, 'is_canonical', 'Ensembl Canonical')]),
        'transcript_attrib': (['transcript_id', 'attrib_type_id', 'value'], [(100, 554, '1')]),
    }  # fmt: skip
    core = write_dump(
        tmp_path,
        {name: [cells(*row) for row in rows] for name, (_, rows) in tables.items()},
        ddl({name: columns for name, (columns, _) in tables.items()}),
    )
    assert loftee_models(core, '1') == {
        'T_PLUS': TranscriptModel(1, 'protein_coding', ((101, 200), (301, 400), (501, 700)), 151, 650),
        'T_MINUS': TranscriptModel(-1, 'protein_coding', ((900, 950), (801, 850)), 811, 940),
        'T_NONCODING': TranscriptModel(1, 'lncRNA', ((1001, 1100),), None, None),
    }
