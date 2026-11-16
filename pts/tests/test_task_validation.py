"""The validation phase every pts task shares: outputs against their dataset models, then the module's check."""

from inspect import unwrap
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import Mock

import polars as pl
import pytest
from otter.config.model import Config
from otter.util.errors import TaskValidationError

from pts import task_validation
from pts.schemas.dataset import DatasetModel, to_polars_schema
from pts.schemas.dataset.discovery import datasets_by_location
from pts.schemas.outputs.biosample import Biosample
from pts.task_validation import model_for, validate_outputs, validate_task
from pts.tasks.pyspark import Pyspark, PysparkSpec


class _Intermediate(DatasetModel):
    dataset_name: ClassVar[str] = '_intermediate'
    dataset_title: ClassVar[str] = 'intermediate'
    dataset_description: ClassVar[str] = 'a dataset validated where it is written, under intermediate/'
    dataset_path: ClassVar[str] = 'intermediate/validated/_intermediate'
    example: ClassVar[dict] = {'id': 'a'}

    id: str


@pytest.fixture
def config(tmp_path: Path) -> Config:
    return Config(step='test', steps=['test'], work_path=tmp_path)


def _write_biosample(path: Path, rows: int = 1) -> str:
    path.mkdir(parents=True)
    pl.DataFrame([Biosample.example] * rows, schema=to_polars_schema(Biosample)).write_parquet(path / 'part.parquet')
    return str(path)


def test_a_model_is_found_by_where_it_is_written(tmp_path: Path, config: Config) -> None:
    assert model_for(str(tmp_path / 'output/biosample'), config) is Biosample
    assert model_for(f'{tmp_path}/output/biosample/', config) is Biosample


@pytest.mark.parametrize('path', ['excluded/biosample', 'intermediate/biosample', 'output/no_such_dataset'])
def test_a_path_no_model_is_written_at_has_none(tmp_path: Path, config: Config, path: str) -> None:
    assert model_for(str(tmp_path / path), config) is None


def test_a_path_outside_the_release_has_no_model(config: Config) -> None:
    assert model_for('/elsewhere/output/biosample', config) is None


def test_a_model_can_be_written_outside_output(tmp_path: Path, config: Config, monkeypatch) -> None:
    """An intermediate dataset is validated by setting `dataset_path`; nothing else changes."""
    monkeypatch.setattr(task_validation, '_models', lambda: {_Intermediate.location(): _Intermediate})

    assert _Intermediate.location() == 'intermediate/validated/_intermediate'
    assert model_for(str(tmp_path / 'intermediate/validated/_intermediate'), config) is _Intermediate


def test_every_model_is_written_at_output_by_default() -> None:
    assert Biosample.location() == 'output/biosample'
    assert all(location.startswith('output/') for location in datasets_by_location())


def test_two_models_at_one_location_are_refused(monkeypatch) -> None:
    class _Twin(_Intermediate):
        dataset_name: ClassVar[str] = '_twin'

    monkeypatch.setattr(
        'pts.schemas.dataset.discovery.discover_datasets', lambda package: {'a': _Intermediate, 'b': _Twin}
    )
    with pytest.raises(ValueError, match='is the location of two models'):
        datasets_by_location()


def test_a_valid_output_passes(tmp_path: Path, config: Config) -> None:
    destination = {'output': _write_biosample(tmp_path / 'output/biosample')}

    assert validate_outputs(destination, config) == []


def test_an_output_breaking_its_model_is_reported(tmp_path: Path, config: Config) -> None:
    destination = _write_biosample(tmp_path / 'output/biosample', rows=2)

    (error,) = validate_outputs(destination, config)
    assert 'breaks its model Biosample' in error
    assert 'unique primary key' in error


def test_an_output_that_was_not_written_is_reported(tmp_path: Path, config: Config) -> None:
    (error,) = validate_outputs(str(tmp_path / 'output/biosample'), config)
    assert 'was not written' in error


def test_destinations_without_a_model_are_not_validated(tmp_path: Path, config: Config) -> None:
    excluded = tmp_path / 'excluded/biosample'
    excluded.mkdir(parents=True)
    (excluded / 'part.parquet').write_text('not parquet at all')

    assert validate_outputs({'excluded': str(excluded), 'other': str(tmp_path / 'intermediate/x')}, config) == []


def test_the_check_runs_with_the_paths_of_the_task(tmp_path: Path, config: Config) -> None:
    check = Mock()
    destination = {'output': _write_biosample(tmp_path / 'output/biosample')}

    validate_task('input/x', destination, {'a': 1}, config, check)

    check.assert_called_once_with('input/x', destination, {'a': 1}, config)


def test_model_and_check_failures_are_reported_together(tmp_path: Path, config: Config) -> None:
    destination = _write_biosample(tmp_path / 'output/biosample', rows=2)
    check = Mock(side_effect=ValueError('3 input rows != 2 written rows'))

    with pytest.raises(TaskValidationError) as raised:
        validate_task('input/x', destination, {}, config, check)
    assert 'unique primary key' in str(raised.value)
    assert '3 input rows != 2 written rows' in str(raised.value)


def test_a_task_without_a_check_or_a_model_is_valid(tmp_path: Path, config: Config) -> None:
    validate_task('input/x', str(tmp_path / 'intermediate/x'), {}, config, None)


def test_a_pyspark_entry_point_without_a_check_has_none() -> None:
    assert Pyspark.load_check('association') is None


def test_pyspark_validate_hands_the_paths_of_the_run_and_the_check_to_validate_task(monkeypatch) -> None:
    """The validation phase is the same as a transform task's."""
    check, validate_task_mock = Mock(), Mock()
    monkeypatch.setattr(Pyspark, 'load_pyspark', staticmethod(lambda _: Mock()))
    monkeypatch.setattr(Pyspark, 'load_check', staticmethod(lambda _: check))
    monkeypatch.setattr('pts.tasks.pyspark.validate_task', validate_task_mock)
    spec = PysparkSpec(
        name='pyspark test', pyspark='association', source='input/x', destination='output/x', settings={'a': 1}
    )
    task = Pyspark(spec, SimpleNamespace(config=Config.model_construct(step='x', work_path='/work')))
    task.srcs, task.dsts = '/work/input/x', '/work/output/x'

    assert unwrap(Pyspark.validate)(task) is task
    validate_task_mock.assert_called_once_with('/work/input/x', '/work/output/x', {'a': 1}, task.context.config, check)
