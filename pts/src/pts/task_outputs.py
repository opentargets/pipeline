"""The validation phase of a task: the datasets its transformer writes, then its module's own rules.

A transformer or pyspark entry point declares the model of each dataset it writes with `writes`:
the model of its single destination, or one model per destination key::

    @writes(evidence=EvidenceEncore)
    def encore(source, destination, settings, config): ...

Each declared dataset is validated against its model (`validate_dataset`). The module
may also define `validate`, taking the transformer's four arguments, for rules no model states; it
runs after the declared datasets.
"""

from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Self, TypeVar

from ot_datamodels import DatasetModel
from otter.config.model import Config
from otter.util.errors import TaskValidationError

from pts.schemas.dataset.validate import validate_dataset
from pts.task_paths import DestinationPaths, SourcePaths

#: Where `writes` keeps a transformer's models, by destination key; `None` keys a single destination.
_WRITES = '__pts_writes__'

#: A module's `validate`: the task's paths, settings and the otter config.
ValidateHook = Callable[[SourcePaths, DestinationPaths, dict[str, Any], Config | None], None]

F = TypeVar('F', bound=Callable[..., None])


def writes(model: type[DatasetModel] | None = None, /, **models: type[DatasetModel]) -> Callable[[F], F]:
    """Declare the model of each dataset a transformer writes.

    Args:
        model: the model of the transformer's single destination.
        **models: the model of each destination key, for a transformer with several.

    Raises:
        TypeError: unless exactly one of the two forms is given.
    """
    if (model is None) == (not models):
        raise TypeError('writes takes the model of a single destination, or models by destination key')
    outputs: dict[str | None, type[DatasetModel]] = {None: model} if model is not None else {**models}

    def declare(transformer: F) -> F:
        setattr(transformer, _WRITES, outputs)
        return transformer

    return declare


@dataclass(frozen=True)
class OutputValidation:
    """What the validation phase of a transformer's tasks checks.

    Attributes:
        name: the transformer's qualified name.
        outputs: the declared models, by destination key; `None` keys a single destination.
        hook: its module's `validate`, if it has one.
    """

    name: str
    outputs: dict[str | None, type[DatasetModel]]
    hook: ValidateHook | None

    @classmethod
    def of(cls, transformer: Callable[..., None], module: ModuleType) -> Self:
        """What `transformer` declares with `writes`, and the `validate` of `module`; either may be absent."""
        return cls(
            f'{transformer.__module__}.{getattr(transformer, "__qualname__", transformer)}',
            getattr(transformer, _WRITES, {}),
            getattr(module, 'validate', None),
        )

    def missing(self, destination: DestinationPaths) -> list[str]:
        """The declared keys a `destination` does not have, `''` standing for a single destination."""
        keys = {key for key, _ in _paths(destination)}
        return sorted(key or '' for key in self.outputs if key not in keys)

    def run(
        self,
        configured: DestinationPaths,
        source: SourcePaths,
        destination: DestinationPaths,
        settings: dict[str, Any],
        config: Config | None,
    ) -> None:
        """Validate a task's written datasets, then run the module's `validate`.

        Args:
            configured: the task's destination as configured, relative to the work path.
            source: the task's resolved source paths.
            destination: the task's resolved destination paths.
            settings: the task's settings.
            config: the otter config.

        Raises:
            TaskValidationError: if a declared destination is not configured, a dataset breaks its
                model, or the module's `validate` fails.
        """
        if missing := self.missing(configured):
            raise TaskValidationError(f'{self.name} declares {missing} with @writes, which the task does not write')
        try:
            for key, model in self.outputs.items():
                validate_dataset(model, _path(destination, key))
            if self.hook is not None:
                self.hook(source, destination, settings, config)
        except ValueError as error:
            raise TaskValidationError(str(error)) from error


def _path(destination: DestinationPaths, key: str | None) -> str:
    if isinstance(destination, dict) and key is not None:
        return destination[key]
    if isinstance(destination, str) and key is None:
        return destination
    raise ValueError(f'no destination for the output {key!r} in {destination}')


def _paths(destination: DestinationPaths) -> list[tuple[str | None, str]]:
    if isinstance(destination, dict):
        return [(key, path) for key, path in destination.items()]
    return [(None, destination)]
