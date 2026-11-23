from __future__ import annotations

import sys
from types import ModuleType

import pytest
from pydantic import BaseModel

from pps.session import Session
from pps.step import InvalidStepError, resolve_step


class ExampleSettings(BaseModel):
    output_path: str


def example_step(session: Session, settings: ExampleSettings) -> None: ...


@pytest.fixture(autouse=True)
def example_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Register `pps.example.steps`, a steps module with one step, `example_step`."""
    module = ModuleType('pps.example.steps')
    module.ExampleSettings = ExampleSettings  # ty: ignore[unresolved-attribute]
    module.example_step = example_step  # ty: ignore[unresolved-attribute]
    monkeypatch.setitem(sys.modules, 'pps.example.steps', module)


def test_resolve_step_returns_function_and_settings_model():
    function, settings_model = resolve_step('example.example_step')

    assert function is example_step
    assert settings_model is ExampleSettings


@pytest.mark.parametrize(
    ('reference', 'message'),
    [
        ('example_step', 'expected <package>.<function>'),
        ('missing_package.step', 'module pps.missing_package.steps not found'),
        ('example.missing_step', 'pps.example.steps.missing_step not found'),
        ('example.ExampleSettings', 'annotated with a pydantic model'),
    ],
)
def test_resolve_step_rejects_invalid_references(reference: str, message: str):
    with pytest.raises(InvalidStepError, match=message):
        resolve_step(reference)
