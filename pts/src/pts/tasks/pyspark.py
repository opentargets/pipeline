"""Task to run a pyspark job."""

from collections.abc import Callable
from importlib import import_module
from typing import Any, Self

from loguru import logger
from otter.config.model import Config
from otter.manifest.model import Artifact
from otter.storage.util import make_absolute
from otter.task.model import Spec, Task, TaskContext
from otter.task.task_reporter import report
from otter.util.errors import TaskValidationError

from pts.task_paths import DestinationPaths, SourcePaths, flatten_paths, resolve_source_paths

PYSPARK_PACKAGE = 'pts.pyspark'

pyspark_entrypoint = Callable[
    [SourcePaths, DestinationPaths, dict[str, Any], dict[str, str]],
    None,
]

#: An entry point module's `validate`: the task's paths, settings and the otter config.
validate_type = Callable[[SourcePaths, DestinationPaths, dict[str, Any], Config], None]


class PysparkSpec(Spec):
    """Specification for a pyspark task."""

    pyspark: str
    """A string with the entry point of the pyspark program.

        The entry point must be a python file in the `pts.pyspark` package, and
        it must contain a method with the same name as the file, which takes
        source, destination and properties as arguments.

        The method should launch a pyspark job to operate on the data read from
        the input paths, and write the result to the destination paths.

        The module may also hold a function named `validate`, next to the entry
        point, taking the source and destination paths, the settings and the otter
        config. It runs in the validation phase of the task: it validates each
        dataset the job writes against its model (`Model.validate_dataset(path)`),
        and checks any further rule, raising `ValueError` when the result is wrong,
        which fails the task. Every dataset written to `output/` must be validated:
        `tests/test_config_dataset_models.py` enforces it for every task in the config.
    """
    source: SourcePaths
    """A path or a dictionary whose values are paths or lists of paths."""
    destination: DestinationPaths
    """A string or a dictionary with the destination paths."""
    settings: dict[str, Any] | None = None
    """A dictionary with settings to pass to the pyspark job."""
    properties: dict[str, str] | None = None
    """A dictionary with the properties to pass to the pyspark job."""


class Pyspark(Task):
    """Task to run a pyspark job."""

    def __init__(
        self,
        spec: PysparkSpec,
        context: TaskContext,
    ) -> None:
        super().__init__(spec, context)
        self.spec: PysparkSpec
        self.pyspark = self.load_pyspark(spec.pyspark)
        self.validate_function = self.load_validate(spec.pyspark)

    @staticmethod
    def load_pyspark(pyspark_name: str) -> pyspark_entrypoint:
        try:
            module = import_module(f'{PYSPARK_PACKAGE}.{pyspark_name}')
            pyspark: pyspark_entrypoint = getattr(module, pyspark_name)
            if not callable(pyspark):
                raise TypeError(f'{pyspark_name} is not a callable')
            return pyspark
        except ImportError:
            raise ModuleNotFoundError(f'{pyspark_name} not found in {PYSPARK_PACKAGE}')

    @staticmethod
    def load_validate(pyspark_name: str) -> validate_type | None:
        """The `validate` function of the entry point's module, if it has one."""
        module = import_module(f'{PYSPARK_PACKAGE}.{pyspark_name}')
        return getattr(module, 'validate', None)

    @report
    def run(self) -> Self:
        self.srcs = resolve_source_paths(self.spec.source, self.context.config)
        self.dsts = make_absolute(self.spec.destination, self.context.config)

        logger.info(f'source paths: {self.srcs}')
        logger.info(f'destination paths: {self.dsts}')
        logger.info(f'settings: {self.spec.settings}')
        logger.info(f'properties: {self.spec.properties}')
        logger.info(f'launching pyspark job: {self.spec.pyspark}')

        self.pyspark(
            self.srcs,
            self.dsts,
            self.spec.settings or {},
            self.spec.properties or {},
        )
        logger.info('pyspark job completed successfully')

        srcs = flatten_paths(self.srcs)
        dsts = list(self.dsts.values()) if isinstance(self.dsts, dict) else self.dsts
        self.artifacts = [Artifact(source=srcs, destination=dsts)]

        return self

    @report
    def validate(self) -> Self:
        """Run the entry point module's `validate`, if it has one; a `ValueError` from it fails the task."""
        if self.validate_function is None:
            return self
        try:
            self.validate_function(self.srcs, self.dsts, self.spec.settings or {}, self.context.config)
        except ValueError as error:
            raise TaskValidationError(str(error)) from error
        return self
