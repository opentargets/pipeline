"""Tests for the search index expression helpers.

Each test named `test_spark_*` pins a spark/polars divergence from the design's divergence
register. They are the reason the port cannot quietly drift back onto a spark assumption.
"""

import polars as pl

from pts.transformers.search.helpers import (
    EMPTY_LIST,
    LIST_STR,
    as_list,
    flatten_cat,
    list_struct_field,
    search_index,
    synonym_field,
    tier,
)


def test_flatten_cat_merges_deduplicates_and_preserves_first_occurrence() -> None:
    frame = pl.DataFrame({'a': [['x', 'y']], 'b': [['y', 'z']]}, schema={'a': LIST_STR, 'b': LIST_STR})

    result = frame.select(flatten_cat(pl.col('a'), pl.col('b')).alias('r'))['r'][0].to_list()

    assert result == ['x', 'y', 'z']


def test_spark_trim_strips_ascii_spaces_only() -> None:
    """Spark's `trim` strips 0x20. polars' bare `strip_chars()` strips all unicode whitespace.

    A non-breaking space inside a synonym must SURVIVE, or the polars index would hold a
    different string than the spark one for every label carrying one.
    """
    frame = pl.DataFrame({'a': [['  padded  ', '\xa0nbsp\xa0']]}, schema={'a': LIST_STR})

    result = frame.select(flatten_cat(pl.col('a')).alias('r'))['r'][0].to_list()

    assert result == ['padded', '\xa0nbsp\xa0']


def test_flatten_cat_strips_every_comma_not_just_the_first() -> None:
    frame = pl.DataFrame({'a': [[' a,b,c ']]}, schema={'a': LIST_STR})

    assert frame.select(flatten_cat(pl.col('a')).alias('r'))['r'][0].to_list() == ['abc']


def test_spark_flatten_cat_drops_a_null_input_list_rather_than_nulling_the_row() -> None:
    """`pl.concat_list` returns NULL if ANY input list is null; spark's flattenCat filters
    the null arrays out and keeps the rest.

    Without the per-input `fill_null`, one null synonym field would wipe out the entire
    keywords array for that disease -- silently, and for tens of thousands of rows.
    """
    frame = pl.DataFrame({'a': [None], 'b': [['kept']]}, schema={'a': LIST_STR, 'b': LIST_STR})

    assert frame.select(flatten_cat(pl.col('a'), pl.col('b')).alias('r'))['r'][0].to_list() == ['kept']


def test_flatten_cat_drops_null_elements_inside_a_list() -> None:
    frame = pl.DataFrame({'a': [['x', None]]}, schema={'a': LIST_STR})

    assert frame.select(flatten_cat(pl.col('a')).alias('r'))['r'][0].to_list() == ['x']


def test_as_list_wraps_a_null_scalar_as_an_empty_contribution() -> None:
    frame = pl.DataFrame({'name': ['a', None]})

    result = frame.select(flatten_cat(as_list(pl.col('name'))).alias('r'))['r'].to_list()

    assert result == [['a'], []]


def test_list_struct_field_extracts_from_a_list_of_structs() -> None:
    schema = {'synonyms': pl.List(pl.Struct({'label': pl.String, 'source': pl.String}))}
    frame = pl.DataFrame({'synonyms': [[{'label': 'L', 'source': 's'}], None]}, schema=schema)

    result = frame.select(flatten_cat(list_struct_field('synonyms', 'label')).alias('r'))['r'].to_list()

    assert result == [['L'], []]


def test_synonym_field_extracts_from_a_struct_of_lists() -> None:
    schema = {'synonyms': pl.Struct({'hasBroadSynonym': LIST_STR, 'hasExactSynonym': LIST_STR})}
    frame = pl.DataFrame({'synonyms': [{'hasBroadSynonym': ['b'], 'hasExactSynonym': None}, None]}, schema=schema)

    result = frame.select(flatten_cat(synonym_field('hasBroadSynonym')).alias('r'))['r'].to_list()

    assert result == [['b'], []]


def test_tier_keeps_only_rows_within_the_rank_limit() -> None:
    frame = pl.DataFrame(
        {'g': ['t'] * 3, 'r': [1, 26, 60], 'v': [['a'], ['b'], ['c']]},
        schema={'g': pl.String, 'r': pl.Int64, 'v': LIST_STR},
    )

    result = frame.group_by('g').agg(tier('v', 'r', 25).alias('t'))['t'][0].to_list()

    assert result == ['a']


def test_spark_tier_drops_null_lists_the_way_collect_list_does() -> None:
    """`collect_list` never collects a null, so a row whose labels are null contributes
    nothing -- it does not null the whole aggregate."""
    frame = pl.DataFrame(
        {'g': ['t'] * 2, 'r': [1, 2], 'v': [['a'], None]},
        schema={'g': pl.String, 'r': pl.Int64, 'v': LIST_STR},
    )

    assert frame.group_by('g').agg(tier('v', 'r', 50).alias('t'))['t'][0].to_list() == ['a']


def test_tier_yields_an_empty_list_not_null_when_nothing_is_in_range() -> None:
    """The distinction matters: the equivalence bar treats null and [] as different values."""
    frame = pl.DataFrame(
        {'g': ['t'], 'r': [99], 'v': [['a']]},
        schema={'g': pl.String, 'r': pl.Int64, 'v': LIST_STR},
    )

    assert frame.group_by('g').agg(tier('v', 'r', 5).alias('t'))['t'][0].to_list() == []


def test_search_index_produces_the_release_schema_in_order() -> None:
    frame = pl.LazyFrame({'k': ['id1'], 'n': ['name1']})

    result = search_index(
        frame,
        id_col=pl.col('k'),
        name_col=pl.col('n'),
        entity='variant',
        category_col=as_list(pl.lit('variant')),
        keywords_col=EMPTY_LIST,
        prefixes_col=EMPTY_LIST,
        ngrams_col=EMPTY_LIST,
    ).collect()

    assert result.columns == [
        'id', 'name', 'description', 'entity', 'category',
        'keywords', 'prefixes', 'ngrams', 'terms', 'terms25', 'terms5', 'multiplier',
    ]
    assert result.schema['multiplier'] == pl.Float64
    assert result.schema['description'] == pl.String
    assert result.row(0) == ('id1', 'name1', None, 'variant', ['variant'], [], [], [], [], [], [], 0.01)
