"""The validation phase of a task: declared datasets against their models, then the module's `validate`."""

from types import ModuleType
from unittest.mock import Mock, call

import pytest
from otter.util.errors import TaskValidationError

from pts.task_outputs import OutputValidation, writes


def _module(**attributes) -> ModuleType:
    module = ModuleType('pts.transformers.fake')
    for name, value in attributes.items():
        setattr(module, name, value)
    return module


def fake(source, destination, settings, config) -> None:
    """A transformer."""


def _model(error: str | None = None) -> Mock:
    return Mock(error=error)


@pytest.fixture(autouse=True)
def validated(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """`validate_dataset`, failing for a model made with an error."""

    def check(model: Mock, path: str) -> None:
        if model.error:
            raise ValueError(model.error)

    mock = Mock(side_effect=check)
    monkeypatch.setattr('pts.task_outputs.validate_dataset', mock)
    return mock


def test_a_transformer_may_declare_nothing() -> None:
    validation = OutputValidation.of(fake, _module())

    assert validation.outputs == {}
    assert validation.hook is None
    validation.run('intermediate/x', '/in', '/work/intermediate/x', {}, None)


def test_a_single_destination_is_validated_against_its_model(validated: Mock) -> None:
    model = _model()
    validation = OutputValidation('fake', {None: model}, None)

    validation.run('output/x', '/in', '/work/output/x', {}, None)

    validated.assert_called_once_with(model, '/work/output/x')


def test_each_declared_destination_is_validated_against_its_model(validated: Mock) -> None:
    first, second = _model(), _model()
    validation = OutputValidation('fake', {'a': first, 'b': second}, None)

    validation.run({'a': 'output/a', 'b': 'output/b'}, '/in', {'a': '/work/output/a', 'b': '/work/output/b'}, {}, None)

    assert validated.mock_calls == [call(first, '/work/output/a'), call(second, '/work/output/b')]


def test_writes_declares_a_single_destination_or_one_per_key() -> None:
    def single(source, destination, settings, config) -> None: ...

    def several(source, destination, settings, config) -> None: ...

    first, second = _model(), _model()

    assert OutputValidation.of(writes(first)(single), _module()).outputs == {None: first}
    assert OutputValidation.of(writes(a=first, b=second)(several), _module()).outputs == {'a': first, 'b': second}


@pytest.mark.parametrize('call', [lambda: writes(), lambda: writes(_model(), a=_model())], ids=['none', 'both'])
def test_writes_takes_one_form(call) -> None:
    with pytest.raises(TypeError, match='single destination, or models by destination key'):
        call()


def test_the_hook_is_the_module_validate() -> None:
    hook = Mock()

    assert OutputValidation.of(fake, _module(validate=hook)).hook is hook


def test_undeclared_destinations_are_not_validated(validated: Mock) -> None:
    model = _model()
    validation = OutputValidation('fake', {'a': model}, None)

    validation.run({'a': 'output/a', 'b': 'output/b'}, '/in', {'a': '/a', 'b': '/b'}, {}, None)

    validated.assert_called_once_with(model, '/a')


def test_a_declared_output_the_task_does_not_write_fails_the_task() -> None:
    validation = OutputValidation('fake', {'a': _model(), 'b': _model()}, None)

    with pytest.raises(TaskValidationError, match=r"declares \['b'\] with @writes"):
        validation.run({'a': 'output/a'}, '/in', {'a': '/a'}, {}, None)


def test_a_dataset_breaking_its_model_fails_the_task() -> None:
    validation = OutputValidation('fake', {None: _model('output/x breaks its model X')}, None)

    with pytest.raises(TaskValidationError, match='breaks its model X'):
        validation.run('output/x', '/in', '/work/output/x', {}, None)


def test_the_module_validate_runs_after_the_models_with_the_task_arguments(validated: Mock) -> None:
    order = Mock()
    model = _model()
    order.attach_mock(validated, 'model')
    order.attach_mock(hook := Mock(), 'hook')
    validation = OutputValidation('fake', {None: model}, hook)

    validation.run('output/x', '/in', '/work/output/x', {'s': 1}, None)

    assert order.mock_calls == [call.model(model, '/work/output/x'), call.hook('/in', '/work/output/x', {'s': 1}, None)]


def test_a_failing_module_validate_fails_the_task() -> None:
    validation = OutputValidation('fake', {}, Mock(side_effect=ValueError('rows went missing')))

    with pytest.raises(TaskValidationError, match='rows went missing'):
        validation.run('intermediate/x', '/in', '/work/intermediate/x', {}, None)
