"""Tests for the polars equivalents of spark's string functions."""

import polars as pl

from pts.transformers.utils.spark_strings import regexp_extract, spark_trim


def test_spark_trim_strips_spaces_only() -> None:
    out = pl.select(spark_trim(pl.lit(pl.Series(['  a b  ', '\ta\t', '\xa0a\xa0', None])))).to_series()
    assert out.to_list() == ['a b', '\ta\t', '\xa0a\xa0', None]


def test_regexp_extract_is_empty_on_no_match_and_null_on_null() -> None:
    value = pl.lit(pl.Series(['PMID:123', 'none', None]))
    assert pl.select(regexp_extract(value, r'PMID:([0-9]+)', 1)).to_series().to_list() == ['123', '', None]
