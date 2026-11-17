"""Explicit validation of what pts tasks write, run by otter in the task's validation phase.

Every transformer (and pyspark entry point) module that writes a dataset to `output/` defines a
function named `validate` next to the transformer, which states the dataset model of each output it
writes::

    from pts.schemas.outputs.evidence_encore import EvidenceEncore
    from pts.task_validation import validate_outputs

    def validate(source: Any, destination: dict[str, str], settings: dict[str, Any], config: Config) -> None:
        validate_outputs({destination['evidence']: EvidenceEncore})

The same function is where any further rule of the task goes -- one no single model can state, such as
rows conserved between the task's inputs and outputs -- raising `ValueError` when it is broken.

`validate_task` runs it, and makes validation mandatory: a task writing to `output/` fails if its module
has no `validate`, or if `validate` does not validate every `output/` destination of the task.
"""

import contextvars
from collections.abc import Callable, Mapping
from typing import Any

from loguru import logger
from otter.config.model import Config
from otter.storage.synchronous.handle import StorageHandle
from otter.util.errors import NotFoundError, StorageError, TaskValidationError

from pts.schemas.dataset import DatasetModel
from pts.schemas.dataset.validate import format_report, validate_dataset
from pts.task_paths import DestinationPaths, SourcePaths

#: A module's `validate`: the task's paths, settings and config; raises `ValueError` when an output is wrong.
Validate = Callable[[SourcePaths, DestinationPaths, dict[str, Any], Config], None]

#: The release-relative prefix of the datasets every task must validate.
OUTPUT_PREFIX = 'output/'

#: The paths `validate_outputs` validated during the current `validate_task`.
_validated: contextvars.ContextVar[list[str] | None] = contextvars.ContextVar('validated', default=None)


def validate_outputs(outputs: Mapping[str, type[DatasetModel]]) -> None:
    """Validate each written dataset against its dataset model.

    Args:
        outputs: dataset paths, each with the model it must satisfy.

    Raises:
        ValueError: listing every dataset that was not written or breaks its model.
    """
    errors = []
    for path, model in outputs.items():
        recorded = _validated.get()
        if recorded is not None:
            recorded.append(path)
        try:
            StorageHandle(path).stat()
        except NotFoundError:
            errors.append(f'{path} was not written, but must satisfy the model {model.__name__}')
            continue
        report = validate_dataset(path, model)
        if not report.is_empty():
            errors.append(f'{path} breaks its model {model.__name__}:\n{format_report(report)}')
    if errors:
        raise ValueError('\n'.join(errors))


def validate_task(
    source: SourcePaths,
    destination: DestinationPaths,
    settings: dict[str, Any],
    config: Config,
    validate: Validate | None,
    module: str,
) -> None:
    """Run a task module's `validate`, and require it to validate every `output/` destination of the task.

    Args:
        source: the task's resolved source paths.
        destination: the task's resolved destination paths.
        settings: the task's settings.
        config: the otter config of the run.
        validate: the module's `validate`, if it has one.
        module: the module's name, for the error messages.

    Raises:
        TaskValidationError: listing the function's failure and every `output/` destination it left unvalidated.
    """
    required = _outputs(destination, config)
    if validate is None:
        if required:
            raise TaskValidationError(
                f'the task writes {sorted(required)}, but {module} defines no validate() to validate them'
            )
        return

    errors = []
    token = _validated.set([])
    try:
        validate(source, destination, settings, config)
    except ValueError as error:
        errors.append(str(error))
    finally:
        validated = {path.rstrip('/') for path in _validated.get() or []}
        _validated.reset(token)

    unvalidated = sorted(required - validated)
    if unvalidated:
        errors.append(f'{module}.validate() did not validate {unvalidated}: every output/ dataset must be validated')
    if errors:
        raise TaskValidationError('\n'.join(errors))
    logger.info(f'{module}.validate() validated {sorted(validated)}')


def _outputs(destination: DestinationPaths, config: Config) -> set[str]:
    """The destinations under `output/` of the release: the ones a task must validate."""
    paths = list(destination.values()) if isinstance(destination, dict) else [destination]
    outputs = set()
    for path in paths:
        try:
            relative = StorageHandle(path, config).relative
        except StorageError:
            continue  # outside the release
        if relative.startswith(OUTPUT_PREFIX):
            outputs.add(path.rstrip('/'))
    return outputs
