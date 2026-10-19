"""The pyspark task accepts the same source shapes as the transform task."""

from inspect import unwrap
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from otter.config.model import Config
from pydantic import ValidationError

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
