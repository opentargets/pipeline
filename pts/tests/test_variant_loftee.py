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

from pts.schemas.vep import vep_schema
from pts.transformers.variant.gerp import KentBigWig
from pts.transformers.variant.loftee import (
    GERP_END_TRUNC_CUTOFF,
    MIN_INTRON_SIZE,
    TranscriptModel,
    add_loftee,
    loftee,
    loftee_models,
)
from pts.transformers.variant.vep import parse_vep
from tests.ensembl_fixtures import cells, ddl, write_dump

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
    """A bgzip FASTA with samtools' .fai and htslib's .gzi, from contig name to (length, start, window)."""
    full = b''.join(b'N' * FASTA_WIDTH + b'\n' for _ in range(BLOCK_BASES // FASTA_WIDTH))
    repeated = bgzf_block(full)
    blocks: list[tuple[int, int]] = []
    fai = []
    compressed = uncompressed = 0
    with path.open('wb') as out:

        def emit(block: bytes, size: int) -> None:
            nonlocal compressed, uncompressed
            blocks.append((compressed, uncompressed))
            out.write(block)
            compressed += len(block)
            uncompressed += size

        for name, (length, start, window) in contigs.items():
            header = f'>{name}\n'.encode()
            emit(bgzf_block(header), len(header))
            fai.append(f'{name}\t{length}\t{uncompressed}\t{FASTA_WIDTH}\t{FASTA_WIDTH + 1}\n')
            for is_full, data in _chunks(length, start, window):
                emit(repeated, len(full)) if is_full else emit(bgzf_block(data), len(data))
        out.write(bgzf_block(b''))
    Path(f'{path}.fai').write_text(''.join(fai))
    gzi = [struct.pack('<Q', len(blocks) - 1)] + [struct.pack('<QQ', c, u) for c, u in blocks[1:]]
    Path(f'{path}.gzi').write_bytes(b''.join(gzi))
    return str(path)


@pytest.fixture(scope='module')
def ancestor(tmp_path_factory: pytest.TempPathFactory) -> str:
    """The reference with the ANCESTRAL bases, on contig 1 only."""
    bases = bytearray(genome())
    for position, base in ANCESTRAL.items():
        bases[position - 1] = ord(base)
    sequence = bases.decode().lower()
    path = tmp_path_factory.mktemp('ancestor') / 'human_ancestor.fa.gz'
    return write_bgzip_fasta(path, {'1': (LENGTH, 1, sequence)})


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
    ancestor: str,
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


def test_stop_gained_far_from_the_end_is_high_confidence(ancestor: str, gerp: KentBigWig):
    assert calls(ancestor, gerp, consequence('TP', ['stop_gained'], 170, 'A', 'T')) == [('HC', None)]


def test_end_trunc(ancestor: str, gerp: KentBigWig):
    # 40 bases from the stop in the last exon, where GERP is -2 a base: -82 <= -58
    assert calls(ancestor, gerp, consequence('TP', ['stop_gained'], 610, 'A', 'T')) == [('LC', 'END_TRUNC')]


def test_end_trunc_needs_low_gerp(ancestor: str, gerp: KentBigWig):
    # the same position on TQ, where GERP is +2 a base
    assert calls(ancestor, gerp, consequence('TQ', ['stop_gained'], 1610, 'A', 'T')) == [('HC', None)]


def test_end_trunc_needs_a_cds_end(ancestor: str, gerp: KentBigWig):
    assert calls(ancestor, gerp, consequence('TP', ['stop_gained'], 610, 'A', 'T', cds_end=None)) == [('HC', None)]


def test_end_trunc_on_an_insertion_inside_the_last_exon(ancestor: str, gerp: KentBigWig):
    # an insertion between 620 and 621 ends before it starts; 30 bases at -2 give -60
    row = consequence('TP', ['frameshift_variant'], 621, '-', 'T', end=620)
    assert calls(ancestor, gerp, row) == [('LC', 'END_TRUNC')]


def test_end_trunc_on_a_deletion(ancestor: str, gerp: KentBigWig):
    assert calls(ancestor, gerp, consequence('TP', ['frameshift_variant'], 611, 'A', '-')) == [('LC', 'END_TRUNC')]


def test_incomplete_cds_is_not_checked(ancestor: str, gerp: KentBigWig):
    # TI's CDS is 103 bases, not whole codons; check_complete_cds is off in production
    assert calls(ancestor, gerp, consequence('TI', ['stop_gained'], 3160, 'A', 'T')) == [('HC', None)]


def test_exon_intron_undef_never_fires(ancestor: str, gerp: KentBigWig):
    # a stop_gained on no exon of TP: LOFTEE only tests for a missing exon number when it has one
    assert calls(ancestor, gerp, consequence('TP', ['stop_gained'], 250, 'A', 'T')) == [('HC', None)]


def test_small_intron(ancestor: str, gerp: KentBigWig):
    assert calls(ancestor, gerp, consequence('TS', ['splice_donor_variant'], 2751, 'G', 'A')) == [
        ('LC', 'SMALL_INTRON')
    ]


def test_non_canonical_splice_is_only_a_flag(ancestor: str, gerp: KentBigWig):
    assert calls(ancestor, gerp, consequence('TQ', ['splice_donor_variant'], 1201, 'A', 'G')) == [('HC', None)]


def test_gc_to_gt_donor(ancestor: str, gerp: KentBigWig):
    assert calls(
        ancestor,
        gerp,
        consequence('TP', ['splice_donor_variant'], 402, 'C', 'T'),
        consequence('TP', ['splice_donor_variant'], 402, 'C', 'A'),
        # on the minus strand the genome's G at 898 is the intron's C
        consequence('TM', ['splice_donor_variant'], 898, 'G', 'A'),
    ) == [('LC', 'GC_TO_GT_DONOR'), ('HC', None), ('LC', 'GC_TO_GT_DONOR')]


def test_utr_splice(ancestor: str, gerp: KentBigWig):
    assert calls(
        ancestor,
        gerp,
        consequence('TU', ['splice_donor_variant'], 2201, 'G', 'A'),
        consequence('TU', ['splice_acceptor_variant'], 2500, 'G', 'A'),
    ) == [('LC', '5UTR_SPLICE'), ('LC', '3UTR_SPLICE')]


def test_anc_allele(ancestor: str, gerp: KentBigWig):
    assert calls(ancestor, gerp, consequence('TP', ['stop_gained'], 160, 'A', 'T')) == [('LC', 'ANC_ALLELE')]


def test_filters_in_plugin_order(ancestor: str, gerp: KentBigWig):
    assert calls(ancestor, gerp, consequence('TP', ['stop_gained'], 600, 'A', 'T')) == [('LC', 'END_TRUNC,ANC_ALLELE')]


def test_anc_allele_only_for_single_base_substitutions(ancestor: str, gerp: KentBigWig):
    # the ancestor has T at 160, but an insertion or an untrimmed pair is never checked
    assert calls(
        ancestor,
        gerp,
        consequence('TP', ['frameshift_variant'], 161, '-', 'T', end=160),
        consequence('TP', ['stop_gained'], 160, 'AA', 'TT'),
    ) == [('HC', None), ('HC', None)]


def test_no_call(ancestor: str, gerp: KentBigWig):
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


def test_chromosome_missing_from_ancestor_and_gerp(ancestor: str, gerp: KentBigWig):
    # MT is in neither file: no GERP and no ancestral allele, so neither END_TRUNC nor ANC_ALLELE
    assert calls(
        ancestor,
        gerp,
        consequence('TP', ['stop_gained'], 600, 'A', 'T'),
        consequence('TP', ['splice_donor_variant'], 402, 'C', 'T'),
        chromosome='MT',
        sequence=b'',
    ) == [('HC', None), ('HC', None)]


def test_order_and_empty_input(ancestor: str, gerp: KentBigWig):
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
    ancestor = write_bgzip_fasta(tmp_path / 'human_ancestor.fa.gz', {'11': (135_086_622, start, ATM['ancestor'])})
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
        other = write_bgzip_fasta(tmp_path / 'derived.fa.gz', {'11': (135_086_622, start, derived)})
        flipped = calls(
            other, gerp, row, chromosome='11', sequence=bytes(sequence), models={ATM['transcript_id']: model}
        )
    assert result == [('HC', None)]
    assert flipped == [('LC', 'ANC_ALLELE')]


def line(position: int, ref: str, alt: str) -> str:
    return '\t'.join(['1', str(position), '.', ref, alt, '.', '.', '.'])


def test_add_loftee_fills_canonical_consequences_for_parse_vep(ancestor: str, gerp: KentBigWig):
    lof = {'canonical': 1, 'gene_id': 'G1', 'biotype': 'protein_coding', 'cds_end': 10}
    records = pl.DataFrame(
        [
            {
                'input': line(610, 'A', 'T'),
                'most_severe_consequence': 'stop_gained',
                'transcript_consequences': [
                    {**lof, 'transcript_id': 'TP', 'consequence_terms': ['stop_gained']},
                    {**lof, 'transcript_id': 'TQ', 'consequence_terms': ['missense_variant']},
                ],
            },
            {'input': line(3000, 'A', 'G'), 'most_severe_consequence': 'intergenic_variant'},
            {
                'input': line(170, 'GA', 'G'),
                'most_severe_consequence': 'frameshift_variant',
                'transcript_consequences': [
                    {**lof, 'transcript_id': 'TP', 'consequence_terms': ['frameshift_variant']},
                    {**lof, 'canonical': None, 'transcript_id': 'TI', 'consequence_terms': ['stop_gained']},
                ],
            },
        ],
        schema=vep_schema,
    ).with_columns(pl.col('transcript_consequences').fill_null(pl.lit([], dtype=vep_schema['transcript_consequences'])))
    filled = add_loftee(records.lazy(), MODELS, genome(), ancestor, gerp, '1').collect()
    assert filled.schema == vep_schema
    assert filled['input'].to_list() == records['input'].to_list()
    lofs = [
        [(tc['transcript_id'], tc['lof'], tc['lof_filter']) for tc in tcs] for tcs in filled['transcript_consequences']
    ]
    assert lofs == [[('TP', 'LC', 'END_TRUNC'), ('TQ', None, None)], [], [('TP', 'HC', None), ('TI', None, None)]]
    index = parse_vep(filled.lazy()).collect()
    # parse_vep keeps canonical transcripts only
    assert [[tc['lofteePrediction'] for tc in tcs] for tcs in index['transcriptConsequences']] == [
        ['LC', None],
        [],
        ['HC'],
    ]
    assert index['variantDescription'][2].endswith(' A high-confidence loss-of-function variant by loftee.')


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
