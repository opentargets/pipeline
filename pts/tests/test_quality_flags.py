"""Tests for splitting flagged datasets into accepted and excluded records."""

from enum import StrEnum

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from pts.transformers.utils.quality_flags import flag_rows, has_excluded_flag, resolve_flag_names, valid_rows


class Flag(StrEnum):
    """Flag names deliberately differ from the persisted flag values."""

    INVALID = 'invalid study'
    OTHER = 'another invalid reason'


def flagged_frame():
    """Retain null and empty lists alongside rows with multiple QC flags."""
    return pl.DataFrame({
        'id': ['null', 'empty', 'valid', 'invalid', 'multiple'],
        'qualityControls': [
            None,
            [],
            ['informational'],
            ['invalid study'],
            ['informational', 'another invalid reason'],
        ],
        'payload': [1, 2, 3, 4, 5],
    }).lazy()


def test_valid_rows_splits_using_enum_values_and_preserves_rows():
    """Configured enum values exclude matching rows without rewriting their contents."""
    frame = flagged_frame()
    result = valid_rows(frame, [Flag.INVALID, Flag.OTHER])
    assert isinstance(result.valid, pl.LazyFrame)
    assert isinstance(result.invalid, pl.LazyFrame)
    assert_frame_equal(result.valid.collect(), frame.filter(pl.col('id').is_in(['null', 'empty', 'valid'])).collect())
    assert_frame_equal(result.invalid.collect(), frame.filter(pl.col('id').is_in(['invalid', 'multiple'])).collect())


def test_valid_rows_excludes_only_selected_reasons():
    """QC reasons outside the exclusion settings do not exclude a row."""
    result = valid_rows(flagged_frame(), [Flag.INVALID])
    assert result.valid.collect()['id'].to_list() == ['null', 'empty', 'valid', 'multiple']
    assert result.invalid.collect()['id'].to_list() == ['invalid']


def test_valid_rows_with_no_invalid_flags_accepts_every_row():
    """An empty reason list accepts all rows, including rows carrying QC flags."""
    frame = flagged_frame()
    result = valid_rows(frame, [])
    assert_frame_equal(result.valid.collect(), frame.collect())
    assert_frame_equal(result.invalid.collect(), frame.collect().head(0))


def test_valid_rows_preserves_schema_for_empty_input():
    """Both empty partitions retain the input schema."""
    frame = flagged_frame().limit(0)
    result = valid_rows(frame, [Flag.INVALID])
    assert_frame_equal(result.valid.collect(), frame.collect())
    assert_frame_equal(result.invalid.collect(), frame.collect())


@pytest.mark.parametrize('flags', [[Flag.INVALID], []], ids=['with-exclusions', 'no-exclusions'])
def test_valid_rows_accepts_untyped_all_empty_quality_controls(flags):
    """A QC column holding only empty and null lists is inferred as List(Null); all rows are valid."""
    frame = pl.LazyFrame({'id': ['empty', 'null'], 'qualityControls': [[], None]})
    assert frame.collect_schema()['qualityControls'] == pl.List(pl.Null)

    result = valid_rows(frame, flags)

    assert result.valid.collect()['id'].to_list() == ['empty', 'null']
    assert result.invalid.collect().is_empty()


def test_flag_rows_appends_unique_sorted_flags_and_initialises_lists():
    frame = pl.LazyFrame({'id': ['a', 'b', 'c', 'd'], 'x': [1, 2, None, 1]})

    once = flag_rows(frame, pl.col('x') == 1, Flag.OTHER)
    twice = flag_rows(once, pl.col('x') >= 1, Flag.INVALID)

    assert twice.collect()['qualityControls'].to_list() == [
        ['another invalid reason', 'invalid study'],
        ['invalid study'],
        [],
        ['another invalid reason', 'invalid study'],
    ]
    assert flag_rows(once, pl.col('x') == 1, Flag.OTHER).collect().equals(once.collect())


def test_flag_rows_accepts_untyped_all_empty_quality_controls():
    frame = pl.LazyFrame({'qualityControls': [[], None], 'x': [1, 0]})

    result = flag_rows(frame, pl.col('x') == 1, Flag.INVALID).collect()

    assert result['qualityControls'].to_list() == [['invalid study'], []]


def test_has_excluded_flag_matches_valid_rows():
    frame = flagged_frame()
    marked = frame.with_columns(has_excluded_flag([Flag.INVALID]).alias('excluded')).collect()
    result = valid_rows(frame, [Flag.INVALID])

    assert marked.filter('excluded')['id'].to_list() == result.invalid.collect()['id'].to_list()


def test_resolve_flag_names_maps_names_and_passes_values_through():
    assert resolve_flag_names(Flag, ['INVALID', 'another invalid reason', Flag.OTHER, 'UNKNOWN']) == [
        Flag.INVALID,
        'another invalid reason',
        Flag.OTHER,
        'UNKNOWN',
    ]
    assert resolve_flag_names(Flag, 'INVALID') == 'INVALID'
