"""Explicit task validation: a module's validate function names each output's model, and must cover every output."""

import re
from inspect import unwrap
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import polars as pl
import pytest
from otter.config.model import Config
from otter.util.errors import TaskValidationError

from pts.pyspark.association import validate as association_validate
from pts.schemas.dataset import to_polars_schema
from pts.schemas.outputs.biosample import Biosample
from pts.task_validation import validate_outputs, validate_task
from pts.tasks.pyspark import Pyspark, PysparkSpec


@pytest.fixture
def config(tmp_path: Path) -> Config:
    return Config(step='test', steps=['test'], work_path=tmp_path)


def _write_biosample(path: Path, rows: int = 1) -> str:
    path.mkdir(parents=True)
    pl.DataFrame([Biosample.example] * rows, schema=to_polars_schema(Biosample)).write_parquet(path / 'part.parquet')
    return str(path)


def test_a_valid_output_passes(tmp_path: Path) -> None:
    validate_outputs({_write_biosample(tmp_path / 'output/biosample'): Biosample})


def test_an_output_breaking_its_model_is_reported(tmp_path: Path) -> None:
    path = _write_biosample(tmp_path / 'output/biosample', rows=2)

    with pytest.raises(ValueError, match=r'breaks its model Biosample:\n.*unique primary key'):
        validate_outputs({path: Biosample})


def test_an_output_that_was_not_written_is_reported(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='was not written'):
        validate_outputs({str(tmp_path / 'output/biosample'): Biosample})


def _validates(outputs: dict[str, type]) -> Mock:
    """A module validate function validating `outputs`."""
    return Mock(side_effect=lambda *_: validate_outputs(outputs))


def test_a_validate_function_covering_every_output_passes(tmp_path: Path, config: Config) -> None:
    output = _write_biosample(tmp_path / 'output/biosample')
    destination = {'output': output, 'excluded': str(tmp_path / 'excluded/biosample')}
    validate = _validates({output: Biosample})

    validate_task('input/x', destination, {'a': 1}, config, validate, 'pts.transformers.biosample')

    validate.assert_called_once_with('input/x', destination, {'a': 1}, config)


def test_a_task_writing_output_without_a_validate_function_fails(tmp_path: Path, config: Config) -> None:
    output = _write_biosample(tmp_path / 'output/biosample')

    with pytest.raises(TaskValidationError, match=r'pts\.transformers\.biosample defines no validate\(\)'):
        validate_task('input/x', output, {}, config, None, 'pts.transformers.biosample')


def test_a_task_writing_no_output_needs_no_validate_function(tmp_path: Path, config: Config) -> None:
    validate_task('input/x', str(tmp_path / 'intermediate/x'), {}, config, None, 'x_validate')


def test_an_output_the_validate_function_leaves_out_fails_the_task(tmp_path: Path, config: Config) -> None:
    validated = _write_biosample(tmp_path / 'output/biosample')
    forgotten = _write_biosample(tmp_path / 'output/other')

    with pytest.raises(TaskValidationError, match=f'did not validate .*{re.escape(forgotten)}'):
        validate_task('input/x', {'a': validated, 'b': forgotten}, {}, config, _validates({validated: Biosample}), 'v')


def test_further_rules_are_reported_with_the_model_failures(tmp_path: Path, config: Config) -> None:
    output = _write_biosample(tmp_path / 'output/biosample', rows=2)

    def validate(*_) -> None:
        try:
            validate_outputs({output: Biosample})
        finally:
            raise ValueError('3 input rows != 2 written rows')

    with pytest.raises(TaskValidationError) as raised:
        validate_task('input/x', output, {}, config, validate, 'v')
    assert '3 input rows != 2 written rows' in str(raised.value)
    assert 'did not validate' not in str(raised.value)


def test_a_pyspark_entry_point_has_its_validate_function_loaded() -> None:
    assert Pyspark.load_validate('association') is association_validate
    assert Pyspark.load_validate('literature_publication_match') is None


def test_pyspark_validate_hands_the_paths_of_the_run_and_the_function_to_validate_task(monkeypatch) -> None:
    """The validation phase is the same as a transform task's."""
    validate, validate_task_mock = Mock(), Mock()
    monkeypatch.setattr(Pyspark, 'load_pyspark', staticmethod(lambda _: Mock()))
    monkeypatch.setattr(Pyspark, 'load_validate', staticmethod(lambda _: validate))
    monkeypatch.setattr('pts.tasks.pyspark.validate_task', validate_task_mock)
    spec = PysparkSpec(
        name='pyspark test', pyspark='association', source='input/x', destination='output/x', settings={'a': 1}
    )
    task = Pyspark(spec, SimpleNamespace(config=Config.model_construct(step='x', work_path='/work')))
    task.srcs, task.dsts = '/work/input/x', '/work/output/x'

    assert unwrap(Pyspark.validate)(task) is task
    validate_task_mock.assert_called_once_with(
        '/work/input/x', '/work/output/x', {'a': 1}, task.context.config, validate, 'pts.pyspark.association'
    )
