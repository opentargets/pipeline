"""Validate datasets pts does not write against their models, in a step of their own.

A pts task validates the datasets it writes itself (`pts.task_validation`). A dataset written outside
pts -- colocalisation, by gentropy -- is validated by a `transform` task with this transformer, reading
it as a source and writing nothing::

    - name: transform schema_validation colocalisation
      transformer: schema_validation
      source:
        colocalisation: output/colocalisation
      destination: {}

Each source is named after the `dataset_name` of the model to validate it against. The transformer does
nothing when the task runs; `schema_validation_check` validates the sources in the task's validation
phase, so a violation fails the task as any other model violation does.
"""

from typing import Any

from otter.config.model import Config

from pts.schemas.dataset import DatasetModel
from pts.schemas.dataset.discovery import discover_datasets
from pts.schemas.dataset.validate import format_report, validate_dataset


def schema_validation(
    source: dict[str, str],
    destination: dict[str, str],
    settings: dict[str, Any],
    config: Config,
) -> None:
    """Check the sources name modelled datasets; the validation itself runs in `schema_validation_check`.

    Raises:
        ValueError: if a source is not a single dataset, a destination is given, or no model has a
            source's name.
    """
    _models_of(source)
    if destination:
        raise ValueError(f'schema_validation writes nothing; its destination must be empty, got {destination!r}')


def schema_validation_check(
    source: dict[str, str],
    destination: dict[str, str],
    settings: dict[str, Any],
    config: Config,
) -> None:
    """Validate each source against the model of the same name.

    Raises:
        ValueError: listing the broken rules of every source that breaks its model.
    """
    errors = []
    for name, model in _models_of(source).items():
        report = validate_dataset(source[name], model)
        if not report.is_empty():
            errors.append(f'{source[name]} breaks its model {model.__name__}:\n{format_report(report)}')
    if errors:
        raise ValueError('\n'.join(errors))


def _models_of(source: dict[str, str]) -> dict[str, type[DatasetModel]]:
    if not isinstance(source, dict) or not all(isinstance(path, str) for path in source.values()):
        raise ValueError(f'schema_validation takes one path per dataset, keyed by dataset name; got {source!r}')
    models = discover_datasets()
    unknown = sorted(set(source) - set(models))
    if unknown:
        raise ValueError(f'no dataset model named {unknown}; modelled datasets: {sorted(models)}')
    return {name: models[name] for name in source}
