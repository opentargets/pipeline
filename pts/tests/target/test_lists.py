"""The list helpers: union, tag and flatten_distinct."""

from __future__ import annotations

import polars as pl

from pts.transformers.target.lists import flatten_distinct, tag, union

LISTS = pl.DataFrame(
    {
        'k': [1, 2, 3, 4],
        'a': [['x', 'y', 'x'], None, ['z'], []],
        'b': [['y', 'w'], ['v'], None, []],
    },
    schema={'k': pl.Int64, 'a': pl.List(pl.String), 'b': pl.List(pl.String)},
)

GROUPED = pl.DataFrame(
    {'g': ['A', 'A', 'A', 'B', 'C'], 'l': [['1', '2'], ['1', '2'], ['2', '3'], None, []]},
    schema={'g': pl.String, 'l': pl.List(pl.String)},
)


def test_union_of_several_lists_keeps_first_occurrences() -> None:
    result = LISTS.select('k', union(pl.col('a'), pl.col('b')).alias('u'))
    assert result['u'].to_list() == [['x', 'y', 'w'], ['v'], ['z'], []]


def test_union_of_one_list_only_fills_null() -> None:
    result = LISTS.select(union(pl.col('a')).alias('u'))
    assert result['u'].to_list() == [['x', 'y', 'x'], [], ['z'], []]


def test_tag_wraps_elements_and_keeps_null() -> None:
    result = LISTS.select(tag(pl.col('b'), 'label', 'HGNC').alias('t'))
    assert result['t'].to_list() == [
        [{'label': 'y', 'source': 'HGNC'}, {'label': 'w', 'source': 'HGNC'}],
        [{'label': 'v', 'source': 'HGNC'}],
        None,
        [],
    ]


def test_flatten_distinct_concatenates_distinct_lists_per_key() -> None:
    result = flatten_distinct(GROUPED, 'g', 'l')
    assert result.to_dicts() == [
        {'g': 'A', 'l': ['1', '2', '2', '3']},
        {'g': 'B', 'l': []},
        {'g': 'C', 'l': []},
    ]
