"""Tests for GERP and AlphaMissense filled into the VEP-shaped records."""

from __future__ import annotations

import gzip
from pathlib import Path

import polars as pl
import pyBigWig  # ty: ignore[unresolved-import] -- compiled module without stubs
import pytest

from pts.schemas.vep import vep_schema
from pts.transformers.variant.predictions import add_gerp_and_alphamissense

SAMPLE = Path(__file__).parent / 'test_data' / 'alphamissense_sample.tsv'


@pytest.fixture(scope='module')
def tsv(tmp_path_factory: pytest.TempPathFactory) -> str:
    path = tmp_path_factory.mktemp('alphamissense') / 'AlphaMissense_hg38.tsv.gz'
    path.write_bytes(gzip.compress(SAMPLE.read_bytes()))
    return str(path)


@pytest.fixture(scope='module')
def bigwig(tmp_path_factory: pytest.TempPathFactory) -> str:
    """A bigWig of contig 1 that scores 945517 to 945519 and the two bases around 946175."""
    path = str(tmp_path_factory.mktemp('gerp') / 'gerp.bw')
    handle = pyBigWig.open(path, 'w')
    handle.addHeader([('1', 1_000_000)])
    scores = {945517: -0.466, 945518: -1.75, 945519: -1.87, 946174: 0.12349, 946175: 0.0625}
    handle.addEntries(
        ['1'] * len(scores),
        [position - 1 for position in scores],
        ends=list(scores),
        values=list(scores.values()),
    )
    handle.close()
    return path


def records(*lines: tuple[str, list[str]]) -> pl.LazyFrame:
    """One record per `input` line, with a transcript consequence for each transcript id."""
    return pl.DataFrame(
        [
            {'input': line, 'transcript_consequences': [{'transcript_id': t} for t in transcripts]}
            for line, transcripts in lines
        ],
        schema={name: vep_schema[name] for name in ('input', 'transcript_consequences')},
    ).lazy()


def line(position: int, ref: str, alt: str) -> str:
    return '\t'.join(['1', str(position), '.', ref, alt, '.', '.', '.'])


def test_fill_conservation_and_alphamissense(bigwig: str, tsv: str):
    found = add_gerp_and_alphamissense(
        records(
            (line(945518, 'C', 'A'), ['ENST00000327044', 'ENST00000999999']),
            (line(946175, 'G', 'A'), ['ENST00000327044']),
        ),
        bigwig,
        tsv,
        '1',
    ).collect()
    first, second = (row['transcript_consequences'] for row in found.iter_rows(named=True))
    assert [(tc['transcript_id'], tc['conservation'], tc['alphamissense']) for tc in first] == [
        ('ENST00000327044', -1.75, {'am_class': 'likely_benign', 'am_pathogenicity': 0.1719}),
        ('ENST00000999999', -1.75, None),
    ]
    assert [(tc['conservation'], tc['alphamissense']) for tc in second] == [
        (0.062, {'am_class': 'likely_benign', 'am_pathogenicity': 0.1618})
    ]


def test_fill_deletion_and_insertion_spans(bigwig: str, tsv: str):
    found = add_gerp_and_alphamissense(
        records(
            (line(945517, 'TCA', 'T'), ['ENST00000327044']),
            (line(945518, 'C', 'CT'), ['ENST00000327044']),
        ),
        bigwig,
        tsv,
        '1',
    ).collect()
    scores = [row['transcript_consequences'][0]['conservation'] for row in found.iter_rows(named=True)]
    # the deletion removes 945518 and 945519; the insertion sits between 945518 and 945519
    assert scores == [-1.75, -1.75]
    assert all(row['transcript_consequences'][0]['alphamissense'] is None for row in found.iter_rows(named=True))


def test_fill_leaves_missing_data_null(bigwig: str, tsv: str):
    found = add_gerp_and_alphamissense(
        records((line(500, 'C', 'A'), ['ENST00000327044']), (line(700, 'G', 'T'), [])),
        bigwig,
        tsv,
        '1',
    ).collect()
    assert found['transcript_consequences'].to_list() == [
        [{**found['transcript_consequences'][0][0], 'conservation': None, 'alphamissense': None}],
        [],
    ]


def test_fill_chromosome_without_contig_or_rows(bigwig: str, tsv: str):
    found = add_gerp_and_alphamissense(
        records((line(945518, 'C', 'A'), ['ENST00000327044'])), bigwig, tsv, 'MT'
    ).collect()
    tc = found['transcript_consequences'][0][0]
    assert (tc['conservation'], tc['alphamissense']) == (None, None)
