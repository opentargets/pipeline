"""The pyspark task accepts the same source shapes as the transform task."""

from inspect import unwrap
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from ot_datamodels.outputs.literature_vector import LiteratureVector
from otter.config.model import Config
from pydantic import ValidationError

from pts.task_outputs import OutputValidation
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
    # `variant` is a placeholder entry point: neither the job nor its outputs are loaded from pts.pyspark
    monkeypatch.setattr(Pyspark, 'load_output_validation', staticmethod(lambda _: Mock()))
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


def test_an_entry_point_has_its_outputs_loaded() -> None:
    assert Pyspark.load_output_validation('literature_vector').outputs == {'vectors': LiteratureVector}
    assert Pyspark.load_output_validation('literature_publication_match') == OutputValidation(
        'pts.pyspark.literature_publication_match.literature_publication_match', {}, None
    )


def test_validate_is_the_same_as_a_transform_task(monkeypatch) -> None:
    validation = Mock()
    monkeypatch.setattr(Pyspark, 'load_pyspark', staticmethod(lambda _: Mock()))
    monkeypatch.setattr(Pyspark, 'load_output_validation', staticmethod(lambda _: validation))
    spec = PysparkSpec(
        name='pyspark test', pyspark='literature_vector', source='input/x', destination='output/x', settings={'a': 1}
    )
    task = Pyspark(spec, SimpleNamespace(config=Config.model_construct(step='x', work_path='/work')))
    task.srcs, task.dsts = '/work/input/x', '/work/output/x'

    assert unwrap(Pyspark.validate)(task) is task
    validation.run.assert_called_once_with('output/x', '/work/input/x', '/work/output/x', {'a': 1}, task.context.config)
