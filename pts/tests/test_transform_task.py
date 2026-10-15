"""Source path validation and resolution for transformer tasks."""

from inspect import unwrap
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from otter.config.model import Config
from pydantic import ValidationError

from pts.tasks.transform import Transform, TransformSpec
from pts.transformers.evidence.encore import encore
from pts.transformers.evidence.reactome import reactome as reactome_evidence
from pts.transformers.reactome import reactome


@pytest.fixture
def doctest_namespace():
    """These task tests do not require the shared Spark doctest session."""
    return {}


@pytest.mark.parametrize('source', ['input/study', {'study': 'input/study'}, {'study': ['input/a', 'input/b']}])
def test_source_shapes(source):
    spec = TransformSpec(name='transform test', transformer='study', source=source, destination='output/study')
    assert spec.source == source


@pytest.mark.parametrize('source', [{'study': [123]}, {'study': {'nested': 'input/a'}}, {'study': [['input/a']]}])
def test_invalid_source_shapes(source):
    with pytest.raises(ValidationError):
        TransformSpec(name='transform test', transformer='study', source=source, destination='output/study')


def test_destination_rejects_lists():
    with pytest.raises(ValidationError):
        TransformSpec(
            name='transform test', transformer='study', source='input/study', destination={'output': ['output/a']}
        )


def test_run_resolves_source_lists_and_flattens_artifacts(tmp_path, monkeypatch):
    transformer = Mock()
    monkeypatch.setattr(Transform, 'load_transformer', staticmethod(lambda _: transformer))
    spec = TransformSpec(
        name='transform test',
        transformer='study',
        source={'study': ['input/a', 'input/b'], 'target': 'input/target', 'optional': []},
        destination={'output': 'output/study', 'excluded': 'excluded/study'},
    )
    config = Config.model_construct(step='study', work_path=str(tmp_path))
    task = Transform(spec, SimpleNamespace(config=config))
    prepare_dirs = Mock()
    monkeypatch.setattr(task, '_prepare_dirs', prepare_dirs)
    unwrap(Transform.run)(task)
    sources = {
        'study': [str(tmp_path / 'input/a'), str(tmp_path / 'input/b')],
        'target': str(tmp_path / 'input/target'),
        'optional': [],
    }
    destinations = {'output': str(tmp_path / 'output/study'), 'excluded': str(tmp_path / 'excluded/study')}
    transformer.assert_called_once_with(sources, destinations, {}, config)
    prepare_dirs.assert_called_once_with(destinations)
    assert task.artifacts is not None
    assert task.artifacts[0].source == [*sources['study'], sources['target']]
    assert task.artifacts[0].destination == list(destinations.values())


def test_loads_a_plain_transformer_name() -> None:
    assert Transform.load_transformer('reactome') is reactome


def test_loads_a_dotted_transformer_name_from_a_subpackage() -> None:
    assert Transform.load_transformer('evidence.encore') is encore


def test_a_plain_name_is_not_shadowed_by_an_evidence_module_of_the_same_name() -> None:
    assert Transform.load_transformer('reactome') is reactome
    assert Transform.load_transformer('evidence.reactome') is reactome_evidence
    assert reactome_evidence is not reactome


def test_raises_for_an_unknown_transformer() -> None:
    with pytest.raises(ModuleNotFoundError):
        Transform.load_transformer('does_not_exist')
