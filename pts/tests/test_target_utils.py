"""Tests for target lookup transformations."""

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from pts.transformers.utils.target import tss_lut


def target_frame(rows):
    """Build targets with explicit nested dtypes, including all-null cases."""
    return pl.DataFrame(
        rows,
        schema={
            'id': pl.String,
            'tss': pl.Int64,
            'canonicalTranscript': pl.Struct({'strand': pl.String, 'start': pl.Int64, 'end': pl.Int64}),
            'genomicLocation': pl.Struct({'strand': pl.Int64, 'start': pl.Int64, 'end': pl.Int64}),
        },
    ).lazy()


@pytest.mark.parametrize(
    ('preferred', 'ct_strand', 'ct_start', 'ct_end', 'gl_strand', 'expected'),
    [
        (9, '+', 100, 200, 1, 9),
        (None, '+', 100, 200, 1, 100),
        (None, '-', 100, 200, 1, 200),
        (None, '+', None, 200, 1, 300),
        (None, '?', 100, 200, -1, 400),
        (None, None, None, None, 1, 300),
        (None, None, None, None, 0, None),
    ],
)
def test_tss_priority(preferred, ct_strand, ct_start, ct_end, gl_strand, expected):
    """Preferred, canonical and genomic coordinates follow strand-specific priority."""
    target = target_frame([
        {
            'id': 'ENSG1',
            'tss': preferred,
            'canonicalTranscript': {'strand': ct_strand, 'start': ct_start, 'end': ct_end},
            'genomicLocation': {'strand': gl_strand, 'start': 300, 'end': 400},
        }
    ])
    result = tss_lut(target)
    assert isinstance(result, pl.LazyFrame)
    assert_frame_equal(
        result.collect(),
        pl.DataFrame({'geneId': ['ENSG1'], 'tss': [expected]}, schema={'geneId': pl.String, 'tss': pl.Int64}),
    )


def test_tss_preserves_duplicate_targets_and_null_structs():
    """Lookup must retain duplicate records and null coordinates."""
    target = target_frame([{'id': 'ENSG1', 'tss': None, 'canonicalTranscript': None, 'genomicLocation': None}] * 2)
    assert tss_lut(target).collect().to_dicts() == [{'geneId': 'ENSG1', 'tss': None}] * 2


def test_empty_tss_lookup_schema():
    """Empty input still yields the lookup contract."""
    assert tss_lut(target_frame([])).collect_schema() == pl.Schema({'geneId': pl.String, 'tss': pl.Int64})


@pytest.mark.parametrize(('preferred', 'strand', 'expected'), [(9, 1, 9), (None, 1, 300), (None, -1, 400)])
def test_tss_without_canonical_transcript(preferred, strand, expected):
    target = target_frame([
        {
            'id': 'ENSG1',
            'tss': preferred,
            'canonicalTranscript': None,
            'genomicLocation': {'strand': strand, 'start': 300, 'end': 400},
        }
    ]).drop('canonicalTranscript')
    assert tss_lut(target).collect().to_dicts() == [{'geneId': 'ENSG1', 'tss': expected}]
