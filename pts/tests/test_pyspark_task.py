"""The pyspark task accepts the same source shapes as the transform task."""

from inspect import unwrap
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from otter.config.model import Config
from otter.util.errors import TaskValidationError
from pydantic import ValidationError

from pts.pyspark.literature_vector import validate as literature_vector_validate
from pts.tasks.pyspark import Pyspark, PysparkSpec


@pytest.mark.parametrize('source', ['input/a', {'a': 'input/a'}, {'a': ['input/a', 'input/b']}])
def test_source_shapes(source):
    spec = PysparkSpec(name='pyspark test', pyspark='variant', source=source, destination='output/a')
    assert spec.source == source


def test_destination_rejects_lists():
    with pytest.raises(ValidationError):
        PysparkSpec(name='pyspark test', pyspark='variant', source='input/a', destination={'output': ['output/a']})


def test_run_resolves_source_lists_and_flattens_artifacts(tmp_path, monkeypatch):
    job = Mock()
    monkeypatch.setattr(Pyspark, 'load_pyspark', staticmethod(lambda _: job))
    # `variant` is a placeholder entry point: neither the job nor its validate function is loaded from pts.pyspark
    monkeypatch.setattr(Pyspark, 'load_validate', staticmethod(lambda _: None))
    spec = PysparkSpec(
        name='pyspark test',
        pyspark='variant',
        source={'vep': 'input/vep', 'annotations': ['input/a', 'input/b']},
        destination={'output': 'output/variant'},
    )
    config = Config.model_construct(step='variant', work_path=str(tmp_path))
    task = Pyspark(spec, SimpleNamespace(config=config))
    unwrap(Pyspark.run)(task)
    sources = {
        'vep': str(tmp_path / 'input/vep'),
        'annotations': [str(tmp_path / 'input/a'), str(tmp_path / 'input/b')],
    }
    job.assert_called_once_with(sources, {'output': str(tmp_path / 'output/variant')}, {}, {})
    assert task.artifacts is not None
    assert task.artifacts[0].source == [sources['vep'], *sources['annotations']]


def test_an_entry_point_has_its_validate_function_loaded() -> None:
    assert Pyspark.load_validate('literature_vector') is literature_vector_validate
    assert Pyspark.load_validate('literature_publication_match') is None


def test_validate_runs_the_module_validate_and_fails_the_task_on_a_value_error(monkeypatch) -> None:
    """The validation phase is the same as a transform task's."""
    validate = Mock(side_effect=ValueError('output/x breaks its model X'))
    monkeypatch.setattr(Pyspark, 'load_pyspark', staticmethod(lambda _: Mock()))
    monkeypatch.setattr(Pyspark, 'load_validate', staticmethod(lambda _: validate))
    spec = PysparkSpec(
        name='pyspark test', pyspark='literature_vector', source='input/x', destination='output/x', settings={'a': 1}
    )
    task = Pyspark(spec, SimpleNamespace(config=Config.model_construct(step='x', work_path='/work')))
    task.srcs, task.dsts = '/work/input/x', '/work/output/x'

    with pytest.raises(TaskValidationError, match='breaks its model X'):
        unwrap(Pyspark.validate)(task)
    validate.assert_called_once_with('/work/input/x', '/work/output/x', {'a': 1}, task.context.config)
