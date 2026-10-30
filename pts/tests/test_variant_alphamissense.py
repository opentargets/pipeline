"""Tests for AlphaMissense joined to transcript consequences."""

from __future__ import annotations

import gzip
from pathlib import Path

import polars as pl
import pytest

from pts.transformers.variant.alphamissense import alphamissense

SAMPLE = Path(__file__).parent / 'test_data' / 'alphamissense_sample.tsv'


@pytest.fixture(scope='module')
def tsv(tmp_path_factory: pytest.TempPathFactory) -> str:
    """The sample of real AlphaMissense lines, gzipped as the release is."""
    path = tmp_path_factory.mktemp('alphamissense') / 'AlphaMissense_hg38.tsv.gz'
    path.write_bytes(gzip.compress(SAMPLE.read_bytes()))
    return str(path)


def rows(*values: tuple[int | None, str, str, str]) -> pl.DataFrame:
    return pl.DataFrame(
        values,
        schema={
            'position': pl.Int64,
            'referenceAllele': pl.String,
            'alternateAllele': pl.String,
            'transcript_id': pl.String,
        },
        orient='row',
    )


def test_alphamissense_join_on_transcript(tsv: str):
    found = alphamissense(
        tsv,
        '1',
        rows(
            (945518, 'C', 'A', 'ENST00000327044'),
            (945518, 'C', 'A', 'ENST00000999999'),
            (946175, 'G', 'A', 'ENST00000327044'),
            (69094, 'G', 'T', 'ENST00000335137'),
        ),
    )
    assert found.select('am_class', 'am_pathogenicity').rows() == [
        ('likely_benign', 0.1719),
        (None, None),
        ('likely_benign', 0.1618),
        ('likely_benign', 0.2937),
    ]
    assert found['transcript_id'].to_list() == [
        'ENST00000327044',
        'ENST00000999999',
        'ENST00000327044',
        'ENST00000335137',
    ]


def test_alphamissense_alleles_must_match(tsv: str):
    found = alphamissense(tsv, '1', rows((945518, 'C', 'G', 'ENST00000327044'), (945519, 'C', 'A', 'ENST00000327044')))
    assert found['am_class'].to_list() == [None, None]


def test_alphamissense_matches_two_digit_transcript_versions(tsv: str):
    found = alphamissense(tsv, '1', rows((3426142, 'G', 'C', 'ENST00000270722')))
    assert found.select('am_class', 'am_pathogenicity').row(0) == ('ambiguous', 0.3495)


def test_alphamissense_null_position_matches_nothing(tsv: str):
    found = alphamissense(tsv, '1', rows((None, 'C', 'A', 'ENST00000327044')))
    assert found['am_class'].to_list() == [None]


def test_alphamissense_chromosome_without_rows_is_null(tsv: str):
    found = alphamissense(tsv, 'X', rows((945518, 'C', 'A', 'ENST00000327044')))
    assert found.select('am_class', 'am_pathogenicity').rows() == [(None, None)]
    assert found.height == 1
