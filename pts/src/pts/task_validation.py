"""The validation phase of every pts task, run by otter after the task has written its outputs.

`Transform` and `Pyspark` tasks validate the same way, through `validate_task`:

* every destination the task wrote is validated against the dataset model written there
  (`pts.schemas.dataset.DatasetModel.location`, by default `output/<dataset_name>`), if there is one;
* then the task module's optional `<name>_check` runs: the place for rules a single dataset's model
  cannot state, such as rows conserved between a task's inputs and its outputs.

Every failure is collected before raising, so one run reports all of them. A failure fails the task, and
otter aborts the step.
"""

import functools
from collections.abc import Callable
from typing import Any

from loguru import logger
from otter.config.model import Config
from otter.storage.synchronous.handle import StorageHandle
from otter.util.errors import NotFoundError, StorageError, TaskValidationError

from pts.schemas.dataset import DatasetModel
from pts.schemas.dataset.discovery import datasets_by_location
from pts.schemas.dataset.validate import format_report, validate_dataset
from pts.task_paths import DestinationPaths, SourcePaths

#: A task module's `<name>_check`: same arguments as the task, raising `ValueError` when the result is wrong.
Check = Callable[[SourcePaths, DestinationPaths, dict[str, Any], Config], None]


@functools.cache
def _models() -> dict[str, type[DatasetModel]]:
    return datasets_by_location()


def model_for(path: str, config: Config) -> type[DatasetModel] | None:
    """The dataset model written at `path`, if any: matched on the path relative to the release root."""
    try:
        relative = StorageHandle(path, config).relative
    except StorageError:
        return None  # outside the release, so no model can be written there
    return _models().get(relative.rstrip('/'))


def validate_outputs(destination: DestinationPaths, config: Config) -> list[str]:
    """Validate every destination a dataset model is written at; the broken rules, one entry per dataset."""
    paths = list(destination.values()) if isinstance(destination, dict) else [destination]
    errors = []
    for path in paths:
        model = model_for(path, config)
        if model is None:
            logger.debug(f'no dataset model is written at {path}; not validated')
            continue
        try:
            StorageHandle(path, config).stat()
        except NotFoundError:
            errors.append(f'{path} was not written, but the model {model.__name__} is written there')
            continue
        report = validate_dataset(path, model)
        if not report.is_empty():
            errors.append(f'{path} breaks its model {model.__name__}:\n{format_report(report)}')
    return errors


def validate_task(
    source: SourcePaths,
    destination: DestinationPaths,
    settings: dict[str, Any],
    config: Config,
    check: Check | None,
) -> None:
    """Validate a task's outputs against their models, then run its module's check.

    Raises:
        TaskValidationError: listing every broken model rule and the check's failure, if any.
    """
    errors = validate_outputs(destination, config)
    if check is not None:
        try:
            check(source, destination, settings, config)
        except ValueError as error:
            errors.append(str(error))
    if errors:
        raise TaskValidationError('\n'.join(errors))
