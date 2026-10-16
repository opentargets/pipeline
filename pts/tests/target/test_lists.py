"""List helpers: spark's safe_array_union and flatten(collect_set(...)) semantics."""

from __future__ import annotations

import polars as pl
import pyspark.sql.functions as f

from pts.pyspark.common.utils import safe_array_union
from pts.target.lists import flatten_distinct, tag, union

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


def test_union_matches_spark_safe_array_union(spark) -> None:
    sdf = spark.createDataFrame(LISTS.to_dicts(), 'k long, a array<string>, b array<string>')
    expected = {
        r['k']: (r['two'], r['one'])
        for r in sdf.select(
            'k', safe_array_union(f.col('a'), f.col('b')).alias('two'), safe_array_union(f.col('a')).alias('one')
        ).collect()
    }
    result = LISTS.select('k', union(pl.col('a'), pl.col('b')).alias('two'), union(pl.col('a')).alias('one'))
    assert {r['k']: (r['two'], r['one']) for r in result.to_dicts()} == expected


def test_flatten_distinct_matches_spark_flatten_collect_set(spark) -> None:
    sdf = spark.createDataFrame(GROUPED.to_dicts(), 'g string, l array<string>')
    expected = {
        r['g']: sorted(r['l']) for r in sdf.groupBy('g').agg(f.flatten(f.collect_set('l')).alias('l')).collect()
    }
    assert {r['g']: sorted(r['l']) for r in flatten_distinct(GROUPED, 'g', 'l').to_dicts()} == expected
