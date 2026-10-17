"""Tests for the Spark-compatible CSV reader, each case pinned against Spark's own reader."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from pyspark.sql import SparkSession

from pts.transformers.utils.spark_csv import read_spark_csv

COLUMNS = ['x', 'y', 'z']

#: name -> file text; the first line is the header.
CASES = {
    'plain': 'x,y,z\n1,2,3\n',
    'empty_field': 'x,y,z\n1,,3\n',
    'quoted_empty_field': 'x,y,z\n1,"",3\n',
    'escaped_quote': 'x,y,z\n1,"a\\"b",3\n',
    'escaped_backslash': 'x,y,z\n1,"a\\\\b",3\n',
    'escaped_backslash_before_closing_quote': 'x,y,z\n1,"a\\\\",3\n',
    'escaped_backslash_and_quote': 'x,y,z\n1,"a\\\\\\"b",3\n',
    'backslash_unquoted_is_literal': 'x,y,z\n1,a\\nb,3\n',
    'backslash_unquoted_quote': 'x,y,z\n1,a\\"b,3\n',
    'backslash_unquoted_pair': 'x,y,z\n1,a\\\\b,3\n',
    'backslash_quoted_is_literal': 'x,y,z\n1,"a\\nb",3\n',
    'backslash_before_separator_in_quotes': 'x,y,z\n1,"a\\,b",3\n',
    'backslash_at_line_end_in_quotes': 'x,y,z\n1,"a\\\n',
    'escaped_quote_swallows_closing': 'x,y,z\n1,"a\\",3\n4,5,6\n',
    'doubled_quote_is_not_an_escape': 'x,y,z\n1,"a""b",3\n',
    'triple_quote': 'x,y,z\n1,"a""",3\n',
    'empty_quoted_then_text': 'x,y,z\n1,""b,3\n',
    'text_after_closing_quote': 'x,y,z\n1,"a"b,3\n',
    'text_before_quote': 'x,y,z\n1,a"b",3\n',
    'quote_inside_unquoted': 'x,y,z\n1,a"b,c\n',
    'unescaped_quote_inside_quotes': 'x,y,z\n1,"a"b"c",3\n',
    'unescaped_quote_after_backslash_pair': 'x,y,z\n1,"a\\\\"b",3\n',
    'unescaped_quote_tail_keeps_backslash': 'x,y,z\n1,"a"b\\"c,3\n',
    'unescaped_quote_tail_stops_at_separator': 'x,y,z\n1,"a"b"c,d",3\n',
    'leading_space_before_quote': 'x,y,z\n1, "a",3\n',
    'space_after_closing_quote': 'x,y,z\n1,"a" ,3\n',
    'space_after_closing_quote_at_line_end': 'x,y,z\n1,2,"a"  \n',
    'separator_in_quotes': 'x,y,z\n1,"a,b",3\n',
    'newline_in_quotes_splits_the_record': 'x,y,z\n1,"a\nb",3\n',
    'carriage_return_in_quotes_splits_the_record': 'x,y,z\n1,"a\rb",3\n',
    'unterminated_quote': 'x,y,z\n1,"abc,def,ghi\n4,5,6\n',
    'lone_quote': 'x,y,z\n1,",3\n',
    'lone_quote_at_line_end': 'x,y,z\n1,2,"\n',
    'short_row': 'x,y,z\n1,2\n',
    'long_row': 'x,y,z\n1,2,3,4,5\n',
    'trailing_separator': 'x,y,z\n1,2,\n',
    'all_empty_row': 'x,y,z\n,,\n',
    'all_quoted_empty_row': 'x,y,z\n"","",""\n',
    'crlf': 'x,y,z\r\n1,2,3\r\n4,5,6\r\n',
    'cr': 'x,y,z\r1,2,3\r4,5,6\r',
    'blank_lines': 'x,y,z\n\n1,2,3\n\n\n4,5,6\n',
    'blank_line_before_header': '\nx,y,z\n1,2,3\n',
    'whitespace_only_line': 'x,y,z\n   \n1,2,3\n\t\n4,5,6\n',
    'trailing_spaces_are_kept': 'x,y,z\n1,a  ,  c\n',
    'quoted_space': 'x,y,z\n1," ",3\n',
    'no_final_newline': 'x,y,z\n1,2,3',
    'header_only': 'x,y,z\n',
    'header_only_without_newline': 'x,y,z',
    'hash_is_not_a_comment': 'x,y,z\n#1,2,3\n',
    'null_text_is_text': 'x,y,z\n1,NA,3\n',
    'non_ascii': 'x,y,z\n1,caf\u00e9\N{NO-BREAK SPACE}x,\u0663\n',
}

#: The separator-sensitive cases, run again with tab and with a custom null value.
SEPARATOR_CASES = {'tab_after_closing_quote': 'x,y,z\n1,"a"\t,3\n', 'null_text': 'x,y,z\n1,null,3\n,"null",""\n'}


def _spark_rows(spark: SparkSession, path: Path, **options: str) -> list[list[str | None]]:
    df = spark.read.csv(str(path), header=True, schema='x string, y string, z string', **options)
    return [list(row) for row in df.collect()]


def _write(path: Path, text: str) -> Path:
    path.write_bytes(text.encode('utf-8'))
    return path


@pytest.mark.parametrize('name', CASES)
def test_reads_like_spark(spark: SparkSession, tmp_path: Path, name: str) -> None:
    path = _write(tmp_path / 'in.csv', CASES[name])
    got = read_spark_csv(str(path), new_columns=COLUMNS)
    assert got.columns == COLUMNS
    assert got.rows() == [tuple(r) for r in _spark_rows(spark, path)]


@pytest.mark.parametrize('name', CASES)
def test_reads_tab_separated_like_spark(spark: SparkSession, tmp_path: Path, name: str) -> None:
    path = _write(tmp_path / 'in.tsv', CASES[name].replace(',', '\t'))
    got = read_spark_csv(str(path), separator='\t', new_columns=COLUMNS)
    assert got.rows() == [tuple(r) for r in _spark_rows(spark, path, sep='\t')]


@pytest.mark.parametrize('name', SEPARATOR_CASES)
@pytest.mark.parametrize(('separator', 'null_values'), [(',', None), ('\t', None), (',', ['null']), ('\t', ['null'])])
def test_separator_and_null_value_like_spark(
    spark: SparkSession, tmp_path: Path, name: str, separator: str, null_values: list[str] | None
) -> None:
    path = _write(tmp_path / 'in.csv', SEPARATOR_CASES[name].replace(',', separator))
    options = {'sep': separator} | ({'nullValue': null_values[0]} if null_values else {})
    got = read_spark_csv(str(path), separator=separator, new_columns=COLUMNS, null_values=null_values)
    assert got.rows() == [tuple(r) for r in _spark_rows(spark, path, **options)]


def test_header_names_are_made_safe_like_spark(spark: SparkSession, tmp_path: Path) -> None:
    path = _write(tmp_path / 'in.csv', 'a,b,A,\n1,2,3,4\n5,6\n7,8,9,10,11\n')
    got = read_spark_csv(str(path))
    expected = spark.read.csv(str(path), header=True)
    assert got.columns == expected.columns
    assert got.rows() == [tuple(r) for r in expected.collect()]


def test_utf8_bom_is_ignored(tmp_path: Path) -> None:
    path = _write(tmp_path / 'in.csv', '\ufeffx,y,z\n1,2,3\n')
    assert read_spark_csv(str(path)).columns == COLUMNS


def test_headerless_file_is_read_by_position(spark: SparkSession, tmp_path: Path) -> None:
    path = _write(tmp_path / 'in.csv', '1,2,3\n4,5\n')
    got = read_spark_csv(str(path), has_header=False, new_columns=COLUMNS)
    expected = spark.read.csv(str(path), header=False, schema='x string, y string, z string')
    assert got.rows() == [tuple(r) for r in expected.collect()]


def test_empty_file_has_no_rows(tmp_path: Path) -> None:
    path = _write(tmp_path / 'in.csv', '')
    got = read_spark_csv(str(path), new_columns=COLUMNS)
    assert got.columns == COLUMNS
    assert got.height == 0


def test_all_columns_are_strings(tmp_path: Path) -> None:
    path = _write(tmp_path / 'in.csv', 'x,y,z\n1,2.5,true\n')
    assert set(read_spark_csv(str(path)).dtypes) == {pl.String}
