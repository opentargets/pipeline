"""Tests for SIFT on the variant step's transcript consequences and its variantEffect entry."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from pts.schemas.variant import variant_schema
from pts.schemas.vep import vep_schema
from pts.transformers.variant.index import normalise_score
from pts.transformers.variant.records import CONSEQUENCES_SCHEMA, ChunkFrames, vep_records
from pts.transformers.variant.sift import AMINO_ACIDS, MATRICES, add_sift, sift_matrices, sift_scores, unescape
from pts.transformers.variant.vep import parse_vep
from tests.variant_fixtures import consequences as variant_consequences

#: Ensembl 116 rows for the canonical translations of ENST00000354588, ENST00000524958,
#: ENST00000390321 and ENST00000526893 (see `test_data/sift/README.md`).
DATA = Path(__file__).parent / 'test_data' / 'sift'
SIFT_ATTRIB = 267


def dump_paths(directory: Path) -> tuple[str, ...]:
    return tuple(
        str(directory / name)
        for name in (
            'protein_function_predictions.txt.gz',
            'translation_md5.txt.gz',
            'attrib.txt.gz',
            'Homo_sapiens.GRCh38.pep.all.fa.gz',
        )
    )


@pytest.fixture(scope='module')
def matrices() -> pl.DataFrame:
    return sift_matrices(*dump_paths(DATA))


def escape(blob: bytes) -> bytes:
    """A blob written as the variation dump writes it."""
    return blob.replace(b'\\', b'\\\\').replace(b'\x00', b'\\0').replace(b'\n', b'\\n').replace(b'\t', b'\\t')


def write_gzip(path: Path, content: bytes) -> None:
    path.write_bytes(gzip.compress(content, mtime=0))


def write_dump(directory: Path, peptides: dict[str, str], rows: list[tuple[int, int, bytes]]) -> tuple[str, ...]:
    """A variation dump whose translation md5 ids number the peptides from 1."""
    paths = dump_paths(directory)
    write_gzip(Path(paths[0]), b''.join(b'%d\t%d\t%s\n' % row for row in rows))
    md5s = [hashlib.md5(sequence.encode()).hexdigest() for sequence in peptides.values()]
    write_gzip(Path(paths[1]), ''.join(f'{number}\t{md5}\n' for number, md5 in enumerate(md5s, 1)).encode())
    write_gzip(Path(paths[2]), f'266\t481\tpolyphen_humdiv\n{SIFT_ATTRIB}\t481\tsift\n'.encode())
    fasta = ''.join(
        f'>ENSP{number} pep chromosome:GRCh38:1:1:9:1 gene:ENSG{number}.1 transcript:{transcript_id}.3\n{sequence}\n'
        for number, (transcript_id, sequence) in enumerate(peptides.items(), 1)
    )
    write_gzip(Path(paths[3]), fasta.encode())
    return paths


def matrix(values: dict[tuple[int, str], int], length: int) -> bytes:
    """A gzip prediction matrix holding `values` by (position, residue), no prediction elsewhere."""
    body = b''.join(
        values.get((position, residue), 0xFFFF).to_bytes(2, 'little')
        for position in range(1, length + 1)
        for residue in AMINO_ACIDS
    )
    return gzip.compress(b'VEP' + body, mtime=0)


def consequences(*rows: tuple[str, str | None, int | None]) -> pl.DataFrame:
    return pl.DataFrame(
        rows, schema={'transcript_id': pl.String, 'amino_acids': pl.String, 'protein_end': pl.Int64}, orient='row'
    )


def test_unescape():
    assert unescape(rb'a\0b\nc\td\\e\\0') == b'a\x00b\nc\td\\e\\0'
    with pytest.raises(ValueError, match='unknown escape'):
        unescape(rb'a\rb')


def test_decode_matrix(tmp_path: Path):
    """Top two bits are the prediction, the low ten the score times 1000, and 0xFFFF is none."""
    blob = matrix(
        {(2, 'V'): 1 << 14 | 10, (2, 'W'): 3 << 14, (3, 'A'): 1000, (3, 'C'): 2 << 14 | 50},
        length=3,
    )
    assert b'\x00' in blob
    paths = write_dump(
        tmp_path,
        {'T1': 'MAV', 'T2': 'MKK', 'T3': 'MLL'},
        [(1, 266, b'not a matrix'), (1, SIFT_ATTRIB, escape(blob)), (2, SIFT_ATTRIB, rb'\N')],
    )
    found = sift_matrices(*paths)
    assert found.schema == MATRICES
    assert found.rows() == [('T1', blob)]
    records = consequences(
        ('T1', 'A/V', 2),
        ('T1', 'A/W', 2),
        ('T1', 'V/A', 3),
        ('T1', 'V/C', 3),
        ('T1', 'M/A', 1),
        ('T1', 'A/V', 4),
        ('T1', 'R/*', 2),
        ('T1', 'A', 2),
        ('T1', 'AV/G', 2),
        ('T1', 'A/X', 2),
        ('T1', None, None),
        ('T2', 'K/V', 2),
        ('T3', 'L/V', 2),
    )
    scores = sift_scores(records, found)
    assert scores.rows() == [
        (0.01, 'deleterious'),
        (0.0, 'deleterious_low_confidence'),
        (1.0, 'tolerated'),
        (0.05, 'tolerated_low_confidence'),
        *[(None, None)] * 9,
    ]


def test_matrices_keep_the_wanted_transcripts(matrices: pl.DataFrame):
    assert matrices['transcript_id'].to_list() == [
        'ENST00000354588',
        'ENST00000390321',
        'ENST00000524958',
        'ENST00000526893',
    ]
    wanted = sift_matrices(*dump_paths(DATA), transcript_ids={'ENST00000390321', 'ENST1'})
    assert wanted['transcript_id'].to_list() == ['ENST00000390321']


def consequence_frame(*transcripts: tuple[str, str, str, int]) -> pl.DataFrame:
    """One variant with a canonical missense consequence per (transcript, gene, amino acids, position)."""
    transcript_id, gene_id, amino_acids, position = (list(column) for column in zip(*transcripts, strict=True))
    return variant_consequences(
        row=[0] * len(transcripts),
        transcript_id=transcript_id,
        gene_id=gene_id,
        amino_acids=amino_acids,
        protein_end=position,
        consequence_terms=[['missense_variant']] * len(transcripts),
        biotype=['protein_coding'] * len(transcripts),
    )


def index_of(input_line: str, consequences: pl.DataFrame) -> pl.DataFrame:
    """The variant index of one variant over the consequences."""
    chromosome, position, _, reference, alternate = input_line.split('\t')
    variants = pl.DataFrame({
        'row': [0],
        'chromosome': [chromosome],
        'position': [int(position)],
        'referenceAllele': [reference],
        'alternateAllele': [alternate],
        'vepStart': [int(position)],
        'vepEnd': [int(position)],
        'vepReference': [reference],
        'vepAlternate': [alternate],
        'hgvsg': [None],
        'most_severe_consequence': ['missense_variant'],
        'reported_transcripts': [True],
    })
    return parse_vep(vep_records(ChunkFrames(variants, consequences)).lazy()).collect()


def sift_effect(index: pl.DataFrame) -> dict:
    [effect] = [effect for effect in index['variantEffect'][0] if effect['method'] == 'SIFT']
    return effect


def test_minimum_score_wins(matrices: pl.DataFrame):
    filled = add_sift(
        consequence_frame(
            ('ENST00000524958', 'ENSG00000278615', 'R/W', 54),
            ('ENST00000354588', 'ENSG00000162194', 'A/V', 220),
        ),
        matrices,
    )
    assert filled.columns == [*CONSEQUENCES_SCHEMA.names(), 'sift_score', 'sift_prediction']
    assert filled.select('transcript_id', 'sift_score', 'sift_prediction').rows() == [
        ('ENST00000524958', 0.09, 'tolerated'),
        ('ENST00000354588', 0.01, 'deleterious_low_confidence'),
    ]
    effect = sift_effect(index_of('11\t62664853\t.\tG\tA', filled))
    assert (effect['assessment'], effect['targetId'], effect['assessmentFlag']) == (
        'deleterious_low_confidence',
        'ENSG00000162194',
        None,
    )
    assert effect['score'] == pytest.approx(0.01)
    assert effect['normalisedScore'] == pytest.approx(0.4)


@pytest.mark.parametrize('reverse', [False, True])
def test_tie_break(matrices: pl.DataFrame, reverse: bool):
    """Both canonical transcripts score 0.01: the more severe prediction wins, whatever the order."""
    transcripts = [
        ('ENST00000390321', 'ENSG00000211675', 'P/Q', 35),
        ('ENST00000526893', 'ENSG00000254709', 'P/Q', 143),
    ]
    filled = add_sift(consequence_frame(*(transcripts[::-1] if reverse else transcripts)), matrices)
    scores = dict(filled.select('transcript_id', 'sift_prediction').iter_rows())
    assert scores == {'ENST00000390321': 'deleterious_low_confidence', 'ENST00000526893': 'deleterious'}
    effect = sift_effect(index_of('22\t22895477\t.\tC\tA', filled))
    assert (effect['assessment'], effect['targetId']) == ('deleterious', 'ENSG00000254709')
    assert effect['score'] == pytest.approx(0.01)
    assert effect['normalisedScore'] == pytest.approx(0.9)


def test_effects_accept_an_empty_consequences_frame(matrices: pl.DataFrame):
    found = add_sift(pl.DataFrame(schema=CONSEQUENCES_SCHEMA), matrices)
    assert found.height == 0
    assert found.select('sift_score', 'sift_prediction').schema == pl.Schema({
        'sift_score': pl.Float64,
        'sift_prediction': pl.String,
    })


def vep_record(*transcripts: dict) -> pl.LazyFrame:
    record = {
        'input': '1\t123\t.\tA\tG',
        'most_severe_consequence': 'missense_variant',
        'transcript_consequences': [
            {'canonical': 1, 'consequence_terms': ['missense_variant'], **transcript} for transcript in transcripts
        ],
    }
    return pl.read_ndjson(io.BytesIO(json.dumps(record).encode()), schema=vep_schema).lazy()


def test_equal_scores_and_predictions_break_by_transcript_id():
    """Non-canonical and unscored transcripts never win."""
    index = parse_vep(
        vep_record(
            {'transcript_id': 'ENST3', 'gene_id': 'g3'},
            {'transcript_id': 'ENST2', 'gene_id': 'g2', 'sift_score': 0.2, 'sift_prediction': 'tolerated'},
            {'transcript_id': 'ENST1', 'gene_id': 'g1', 'sift_score': 0.2, 'sift_prediction': 'tolerated'},
            {'transcript_id': 'ENST0', 'gene_id': 'g0', 'sift_score': 0.0, 'sift_prediction': 'deleterious'}
            | {'canonical': None},
        )
    ).collect()
    effect = sift_effect(index)
    assert (effect['assessment'], effect['targetId']) == ('tolerated', 'g1')
    assert effect['score'] == pytest.approx(0.2)


def test_no_sift_no_entry():
    index = parse_vep(vep_record({'transcript_id': 'ENST1', 'gene_id': 'g1'})).collect()
    assert [effect['method'] for effect in index['variantEffect'][0]] == ['VEP']


def test_sift_prediction_not_in_transcripts():
    index = parse_vep(
        vep_record({'transcript_id': 'ENST1', 'gene_id': 'g1', 'sift_score': 0.0, 'sift_prediction': 'deleterious'})
    ).collect()
    assert index.schema == variant_schema
    [transcript] = index['transcriptConsequences'][0]
    assert not [name for name in transcript if 'sift' in name.lower()]
    assert sift_effect(index)['assessment'] == 'deleterious'


def _rescale(tolerance: float, low: float, high: float, minimum: float, maximum: float) -> float:
    return (tolerance - low) / (high - low) * (maximum - minimum) + minimum


@pytest.mark.parametrize(
    ('assessment', 'score', 'expected'),
    [
        ('deleterious', 0.01, 0.9),
        ('deleterious', 0.0, 1.0),
        ('deleterious_low_confidence', 0.01, 0.4),
        ('deleterious_low_confidence', 0.05, 0.0),
        ('tolerated_low_confidence', 0.05, 0.0),
        ('tolerated_low_confidence', 0.5, _rescale(0.5, 0, 0.95, -0.5, 0)),
        ('tolerated', 0.09, _rescale(0.91, 0, 0.95, -1, -0.5)),
        ('tolerated', 1.0, -1.0),
        ('deleterious', 0.2, None),
        ('tolerated', 0.01, None),
        ('unknown', 0.5, None),
        ('tolerated', None, None),
    ],
)
def test_normalisation(assessment: str, score: float | None, expected: float | None):
    frame = pl.DataFrame(
        {'score': [score], 'assessment': [assessment]}, schema={'score': pl.Float32, 'assessment': pl.String}
    )
    value = frame.select(normalise_score('SIFT', pl.col('score'), pl.col('assessment'))).item()
    assert value == (None if expected is None else pytest.approx(expected, abs=1e-6))
    assert expected is None or np.isfinite(value)
