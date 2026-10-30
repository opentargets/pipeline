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
            'genomicLocation': pl.Struct({'strand': pl.Int64, 'start': pl.Int64, 'end': pl.Int64}),
        },
    ).lazy()


@pytest.mark.parametrize(
    ('preferred', 'strand', 'expected'),
    [(9, 1, 9), (None, 1, 300), (None, -1, 400), (None, 0, None), (None, None, None)],
)
def test_tss_priority(preferred, strand, expected):
    """The preferred TSS wins; otherwise the strand-aware genomic start or end."""
    target = target_frame([
        {'id': 'ENSG1', 'tss': preferred, 'genomicLocation': {'strand': strand, 'start': 300, 'end': 400}}
    ])
    result = tss_lut(target)
    assert isinstance(result, pl.LazyFrame)
    assert_frame_equal(
        result.collect(),
        pl.DataFrame({'geneId': ['ENSG1'], 'tss': [expected]}, schema={'geneId': pl.String, 'tss': pl.Int64}),
    )


def test_tss_ignores_canonical_transcript():
    """A canonical transcript (PTS types its strand Int32) is not used for the TSS."""
    target = target_frame([{'id': 'ENSG1', 'tss': None, 'genomicLocation': {'strand': 1, 'start': 300, 'end': 400}}])
    canonical = pl.Struct({'strand': pl.Int32, 'start': pl.Int64, 'end': pl.Int64})
    target = target.with_columns(
        pl.lit({'strand': -1, 'start': 1, 'end': 2}, dtype=canonical).alias('canonicalTranscript')
    )
    assert tss_lut(target).collect().to_dicts() == [{'geneId': 'ENSG1', 'tss': 300}]


def test_tss_preserves_duplicate_targets_and_null_structs():
    """Lookup must retain duplicate records and null coordinates."""
    target = target_frame([{'id': 'ENSG1', 'tss': None, 'genomicLocation': None}] * 2)
    assert tss_lut(target).collect().to_dicts() == [{'geneId': 'ENSG1', 'tss': None}] * 2


def test_empty_tss_lookup_schema():
    """Empty input still yields the lookup contract."""
    assert tss_lut(target_frame([])).collect_schema() == pl.Schema({'geneId': pl.String, 'tss': pl.Int64})
