"""Tests for AlphaMissense joined to transcript consequences."""

from __future__ import annotations

import gzip
from pathlib import Path

import polars as pl
import pytest

from pts.transformers.variant.annotators.alphamissense import add_alphamissense, alphamissense, partition_alphamissense
from pts.transformers.variant.vep.frames import CONSEQUENCES_SCHEMA
from tests.variant_fixtures import consequence_rows

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


def test_partition_reads_the_same_predictions(tsv: str, tmp_path: Path):
    parts = partition_alphamissense(tsv, tmp_path, ['1', 'X'])
    assert parts == {'1': str(tmp_path / '1.parquet'), 'X': str(tmp_path / 'X.parquet')}
    keys = rows((945518, 'C', 'A', 'ENST00000327044'), (3426142, 'G', 'C', 'ENST00000270722'))
    assert alphamissense(parts['1'], '1', keys).equals(alphamissense(tsv, '1', keys))
    assert pl.read_parquet(parts['X']).is_empty()
    assert alphamissense(parts['X'], 'X', keys)['am_class'].to_list() == [None, None]


def test_alphamissense_reads_mt_as_chrm(tsv: str, tmp_path: Path):
    """The file names the mitochondrion `chrM`; MT_12706_T_C is a real prediction."""
    keys = rows((12706, 'T', 'C', 'ENST00000361567'))
    assert alphamissense(tsv, 'MT', keys).select('am_class', 'am_pathogenicity').row(0) == ('likely_pathogenic', 0.9989)
    parts = partition_alphamissense(tsv, tmp_path, ['MT'])
    assert pl.read_parquet(parts['MT']).height == 1
    assert alphamissense(parts['MT'], 'MT', keys).equals(alphamissense(tsv, 'MT', keys))


def predicted(found: pl.DataFrame) -> list[tuple]:
    return found.select('am_class', 'am_pathogenicity').rows()


def test_add_alphamissense_matches_each_consequence_on_its_transcript(tsv: str):
    found = add_alphamissense(
        consequence_rows(
            (0, 945518, 'C', 'A', 'ENST00000327044'),
            (0, 945518, 'C', 'A', 'ENST00000999999'),
            (1, 946175, 'G', 'A', 'ENST00000327044'),
        ),
        tsv,
        '1',
    )
    assert predicted(found) == [('likely_benign', 0.1719), (None, None), ('likely_benign', 0.1618)]


def test_add_alphamissense_skips_indels(tsv: str):
    found = add_alphamissense(
        consequence_rows(
            (0, 945517, 'TCA', 'T', 'ENST00000327044'),
            (1, 945518, 'C', 'CT', 'ENST00000327044'),
        ),
        tsv,
        '1',
    )
    assert predicted(found) == [(None, None), (None, None)]


def test_add_alphamissense_leaves_unpredicted_variants_and_chromosomes_null(tsv: str):
    unpredicted = add_alphamissense(consequence_rows((0, 500, 'C', 'A', 'ENST00000327044')), tsv, '1')
    chromosome = add_alphamissense(consequence_rows((0, 945518, 'C', 'A', 'ENST00000327044')), tsv, 'MT')
    assert predicted(unpredicted) == predicted(chromosome) == [(None, None)]


def test_add_alphamissense_accepts_an_empty_consequences_frame():
    found = add_alphamissense(pl.DataFrame(schema=CONSEQUENCES_SCHEMA), '/no/am.tsv.gz', '1')
    assert found.height == 0
    assert found.select('am_class', 'am_pathogenicity').schema == pl.Schema({
        'am_class': pl.String,
        'am_pathogenicity': pl.Float64,
    })
