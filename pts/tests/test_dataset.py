"""Tests for the shared dataset reader and writer."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import polars as pl
import pytest

from pts.transformers.utils.dataset import read_dataset


def _write_parts(directory: Path, *frames: pl.DataFrame) -> str:
    """Write one parquet part per frame, the way an upstream step lays a dataset out."""
    directory.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(frames):
        frame.write_parquet(directory / f'{index:08d}.parquet')
    return str(directory)


def test_read_parquet_from_a_single_file(tmp_path: Path) -> None:
    path = tmp_path / 'one.parquet'
    pl.DataFrame({'a': [1, 2]}).write_parquet(path)

    assert read_dataset(str(path)).collect().height == 2


def test_read_parquet_from_a_directory_of_parts(tmp_path: Path) -> None:
    path = _write_parts(tmp_path / 'ds', pl.DataFrame({'a': [1]}), pl.DataFrame({'a': [2]}))

    assert sorted(read_dataset(path).collect()['a'].to_list()) == [1, 2]


def test_read_parquet_skips_underscore_prefixed_files(tmp_path: Path) -> None:
    """Spark treats `_`-prefixed files as metadata and skips them; a bare glob does not.

    Measured: planting a `_hidden.parquet` in a real directory made it contribute rows.
    """
    directory = tmp_path / 'ds'
    path = _write_parts(directory, pl.DataFrame({'a': [1]}))
    pl.DataFrame({'a': [999]}).write_parquet(directory / '_hidden.parquet')

    assert read_dataset(path).collect()['a'].to_list() == [1]


def test_read_parquet_raises_when_a_directory_has_no_parts(tmp_path: Path) -> None:
    directory = tmp_path / 'ds'
    directory.mkdir()
    pl.DataFrame({'a': [1]}).write_parquet(directory / '_only_hidden.parquet')

    with pytest.raises(ValueError, match=r'no .* files found'):
        read_dataset(str(directory))


def test_read_ndjson_from_a_gzipped_file(tmp_path: Path) -> None:
    path = tmp_path / 'x.json.gz'
    path.write_bytes(gzip.compress(json.dumps({'a': 'v'}).encode()))

    frame = read_dataset(str(path), format='ndjson').collect()

    assert frame['a'].to_list() == ['v']


def test_read_ndjson_from_a_directory_of_parts(tmp_path: Path) -> None:
    directory = tmp_path / 'lit'
    directory.mkdir()
    for index, value in enumerate(['a', 'b']):
        (directory / f'part-{index}.json.gz').write_bytes(gzip.compress(json.dumps({'v': value}).encode()))

    frame = read_dataset(str(directory), format='ndjson').collect()

    assert sorted(frame['v'].to_list()) == ['a', 'b']


def test_schema_pins_dtypes_and_column_order(tmp_path: Path) -> None:
    """Column order is load-bearing: it feeds de-duplication content hashes downstream."""
    path = tmp_path / 'x.json.gz'
    path.write_bytes(gzip.compress(json.dumps({'z': 1, 'a': 2}).encode()))

    schema = {'a': pl.String, 'z': pl.String}
    frame = read_dataset(str(path), format='ndjson', schema=schema).collect()

    assert frame.columns == ['a', 'z']
    assert frame.dtypes == [pl.String, pl.String]


def test_schema_with_parquet_raises_rather_than_being_ignored(tmp_path: Path) -> None:
    """Silently ignoring a passed schema is the trap this guards.

    Polars would return the file's own dtypes with no error, so a caller pinning a schema
    would never learn it did nothing.
    """
    path = tmp_path / 'x.parquet'
    pl.DataFrame({'a': [1]}).write_parquet(path)

    with pytest.raises(ValueError, match='only applied to ndjson'):
        read_dataset(str(path), schema={'a': pl.String})


def test_unrecognised_format_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='unrecognised format'):
        read_dataset(str(tmp_path), format='csv')
