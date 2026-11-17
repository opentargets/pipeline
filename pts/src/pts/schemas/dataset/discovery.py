"""Find every dataset model without a registry.

Every module under `pts.schemas.outputs` is imported, and every concrete `DatasetModel` subclass
defined there is a dataset. Nothing has to be listed anywhere: adding a module with a model is
enough for the test suite and the documentation to pick it up. Struct models and shared bases in
the same package are ignored, since they are not concrete datasets.
"""

import importlib
import pkgutil

from pts.schemas.dataset.base import DatasetModel

#: The package holding every output dataset model.
OUTPUTS_PACKAGE = 'pts.schemas.outputs'


def discover_datasets(package: str = OUTPUTS_PACKAGE) -> dict[str, type[DatasetModel]]:
    """Every concrete dataset model defined under `package`, by dataset name.

    Raises:
        ValueError: if two models declare the same dataset name.
    """
    root = importlib.import_module(package)
    for module in pkgutil.walk_packages(root.__path__, f'{package}.'):
        importlib.import_module(module.name)

    datasets: dict[str, type[DatasetModel]] = {}
    for model in _subclasses(DatasetModel):
        if not model.is_concrete() or not model.__module__.startswith(f'{package}.'):
            continue
        if model.dataset_name in datasets:
            other = datasets[model.dataset_name]
            msg = f'dataset {model.dataset_name!r} is declared twice: {_qualname(other)} and {_qualname(model)}'
            raise ValueError(msg)
        datasets[model.dataset_name] = model
    return datasets


def _subclasses(cls: type[DatasetModel]) -> list[type[DatasetModel]]:
    """Every subclass of `cls`, however indirect, each once (a class may inherit from two of them)."""
    found: list[type[DatasetModel]] = []
    pending = list(cls.__subclasses__())
    while pending:
        subclass = pending.pop()
        if subclass not in found:
            found.append(subclass)
            pending.extend(subclass.__subclasses__())
    return found


def _qualname(cls: type) -> str:
    return f'{cls.__module__}.{cls.__qualname__}'
