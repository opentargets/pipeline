"""Resolution of step functions referenced from `config.yaml`.

A step is a plain function living in `pps.<package>.steps`, taking a `Session` and
a pydantic settings model:

    def top_hit_ingestion(session: Session, settings: TopHitIngestionSettings) -> None: ...

It is referenced from a task in `config.yaml` as `<package>.<function>`, e.g.
`gwas_catalog.top_hit_ingestion`.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import Any, get_type_hints

from pydantic import BaseModel

from pps.session import Session

STEPS_MODULE = 'steps'

StepFunction = Callable[[Session, Any], None]


class InvalidStepError(ValueError):
    """A step reference does not resolve to a valid step function."""


def resolve_step(reference: str) -> tuple[StepFunction, type[BaseModel]]:
    """Find a step function and its settings model from a `<package>.<function>` reference.

    Raises:
        InvalidStepError: If the reference is malformed, the function does not exist, or its
            signature is not `(session: Session, settings: <BaseModel subclass>)`.
    """
    package, _, function_name = reference.rpartition('.')
    if not package or not function_name:
        raise InvalidStepError(f'invalid step reference {reference!r}: expected <package>.<function>')

    module_name = f'pps.{package}.{STEPS_MODULE}'
    try:
        module = import_module(module_name)
    except ModuleNotFoundError as e:
        raise InvalidStepError(f'invalid step reference {reference!r}: module {module_name} not found') from e

    function = getattr(module, function_name, None)
    if not callable(function):
        raise InvalidStepError(f'invalid step reference {reference!r}: {module_name}.{function_name} not found')

    settings_model = get_type_hints(function).get('settings')
    if not (isinstance(settings_model, type) and issubclass(settings_model, BaseModel)):
        raise InvalidStepError(f'step {reference!r} must take a `settings` argument annotated with a pydantic model')

    return function, settings_model
