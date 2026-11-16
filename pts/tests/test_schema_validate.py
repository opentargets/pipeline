"""Tests for validating written datasets, and for the `schema_validation` task."""

from pathlib import Path
from typing import Annotated, ClassVar

import polars as pl
import pytest
from pydantic import Field

from pts.schemas.dataset import DatasetModel, PrimaryKey, to_polars_schema, validate
from pts.schemas.dataset.validate import DUPLICATE_KEY, main, validate_dataset
from pts.schemas.outputs.biosample import BiosampleIndex
from pts.transformers.schema_validation import schema_validation, schema_validation_check


class _Scores(DatasetModel):
    dataset_name: ClassVar[str] = '_scores'
    dataset_title: ClassVar[str] = 'scores'
    dataset_description: ClassVar[str] = 'scores'
    example: ClassVar[dict] = {'id': 'a', 'score': 0.5}

    id: Annotated[str, PrimaryKey()]
    score: float = Field(ge=0.0, le=1.0)


class _Pairs(DatasetModel):
    dataset_name: ClassVar[str] = '_pairs'
    dataset_title: ClassVar[str] = 'pairs'
    dataset_description: ClassVar[str] = 'pairs'
    example: ClassVar[dict] = {'left': 'a', 'right': 'b'}

    left: Annotated[str, PrimaryKey()]
    right: Annotated[str, PrimaryKey()]


def _write_parts(path: Path, *parts: pl.DataFrame) -> str:
    path.mkdir(parents=True)
    for index, part in enumerate(parts):
        part.write_parquet(path / f'{index:08d}.parquet')
    return str(path)


def _scores(ids: list[str], scores: list[float]) -> pl.DataFrame:
    return pl.DataFrame({'id': ids, 'score': scores})


def test_a_valid_dataset_has_an_empty_report(tmp_path: Path) -> None:
    path = _write_parts(tmp_path / 'ds', _scores(['a', 'b'], [0.1, 1.0]), _scores(['c'], [0.0]))

    assert validate_dataset(path, _Scores).is_empty()


def test_violations_are_counted_across_parts(tmp_path: Path) -> None:
    path = _write_parts(
        tmp_path / 'ds',
        _scores(['a', 'b'], [1.5, 0.1]),
        _scores(['c'], [0.2]),
        _scores(['d', 'e'], [2.0, 1.5]),
    )

    report = validate_dataset(path, _Scores)

    assert report.select('dataset', 'column', 'check', 'failures', 'parts').rows() == [
        ('_scores', 'score', 'score: <= 1.0', 3, 2)
    ]
    assert report['examples'].to_list() == [['1.5', '2.0']]
    assert report['first_part'].item().endswith('00000000.parquet')


def test_missing_and_undeclared_columns_are_reported(tmp_path: Path) -> None:
    path = _write_parts(tmp_path / 'ds', pl.DataFrame({'id': ['a'], 'other': [1]}))

    report = validate_dataset(path, _Scores)

    assert sorted(report['check'].to_list()) == ['column_in_dataframe', 'column_in_schema']


def test_a_key_repeated_in_different_parts_is_a_duplicate(tmp_path: Path) -> None:
    path = _write_parts(tmp_path / 'ds', _scores(['a', 'b'], [0.1, 0.2]), _scores(['b', 'b', 'c'], [0.3, 0.4, 0.5]))

    report = validate_dataset(path, _Scores)

    assert report.select('column', 'check', 'failures', 'parts', 'examples').rows() == [
        ('id', DUPLICATE_KEY, 2, 2, ['b'])
    ]
    assert report['first_part'].item().endswith('00000000.parquet')


def test_a_hash_collision_is_not_a_duplicate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every key hashes alike here, so only the comparison on the key itself tells them apart."""
    monkeypatch.setattr(validate, '_key_hash', lambda key: pl.first(key).str.len_bytes().cast(pl.UInt64))
    path = _write_parts(tmp_path / 'ds', _scores(['a', 'b'], [0.1, 0.2]), _scores(['c', 'c'], [0.3, 0.4]))

    report = validate_dataset(path, _Scores)

    assert report.select('failures', 'examples').rows() == [(1, ['c'])]


def test_a_missing_key_column_is_reported_once(tmp_path: Path) -> None:
    path = _write_parts(tmp_path / 'ds', pl.DataFrame({'score': [0.1, 0.1]}))

    assert validate_dataset(path, _Scores)['check'].to_list() == ['column_in_dataframe']


def test_a_composite_key_is_unique_on_its_columns_together(tmp_path: Path) -> None:
    pairs = pl.DataFrame({'left': ['a', 'a', 'b'], 'right': ['x', 'y', 'x']})
    unique = _write_parts(tmp_path / 'unique', pairs)
    repeated = _write_parts(tmp_path / 'repeated', pairs, pairs.head(1))

    assert validate_dataset(unique, _Pairs).is_empty()
    assert validate_dataset(repeated, _Pairs).select('column', 'failures', 'examples').rows() == [
        ('left, right', 1, ['a, x'])
    ]


def test_a_single_parquet_file_is_validated(tmp_path: Path) -> None:
    path = tmp_path / 'ds.parquet'
    _scores(['a'], [3.0]).write_parquet(path)

    assert validate_dataset(str(path), _Scores)['failures'].to_list() == [1]


def _biosample(tmp_path: Path, rows: int) -> str:
    frame = pl.DataFrame([BiosampleIndex.example] * rows, schema=to_polars_schema(BiosampleIndex))
    return _write_parts(tmp_path / 'biosample', frame)


def test_the_task_writes_an_empty_report_and_passes_its_check(tmp_path: Path) -> None:
    source = {'biosample': _biosample(tmp_path, 1)}
    destination = str(tmp_path / 'report')

    schema_validation(source, destination, {}, None)

    assert pl.read_parquet(destination).is_empty()
    schema_validation_check(source, destination, {}, None)


def test_the_task_writes_the_violations_and_fails_its_check(tmp_path: Path) -> None:
    source = {'biosample': _biosample(tmp_path, 2)}
    destination = str(tmp_path / 'report')

    schema_validation(source, destination, {}, None)

    assert pl.read_parquet(destination).select('dataset', 'check').rows() == [('biosample', DUPLICATE_KEY)]
    with pytest.raises(ValueError, match='1 dataset\\(s\\) break their models'):
        schema_validation_check(source, destination, {}, None)


def test_the_task_refuses_a_dataset_without_a_model(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no dataset model named \\['nonexistent'\\]"):
        schema_validation({'nonexistent': str(tmp_path)}, str(tmp_path / 'report'), {}, None)


def test_the_command_line_validates_an_output_directory(tmp_path: Path) -> None:
    _biosample(tmp_path, 1)

    assert main([str(tmp_path), 'biosample']) == 0
    assert main([str(tmp_path), 'nonexistent']) == 2
    assert main([str(tmp_path / 'empty'), 'biosample']) == 2
