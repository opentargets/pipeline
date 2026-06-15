"""Task to run a pyspark job."""

from collections.abc import Callable
from importlib import import_module
from typing import Any, Self

from loguru import logger
from otter.manifest.model import Artifact
from otter.storage.util import make_absolute
from otter.task.model import Spec, Task, TaskContext
from otter.task.task_reporter import report

PYSPARK_PACKAGE = 'pts.pyspark'

path_or_paths = str | dict[str, str]

pyspark_entrypoint = Callable[
    [path_or_paths, path_or_paths, dict[str, Any], dict[str, str]],
    None,
]


class PysparkSpec(Spec):
    """Specification for a pyspark task."""

    pyspark: str
    """A string with the entry point of the pyspark program.

        The entry point must be a python file in the `pts.pyspark` package, and
        it must contain a method with the same name as the file, which takes
        source, destination and properties as arguments.

        The method should launch a pyspark job to operate on the data read from
        the input paths, and write the result to the destination paths.
    """
    source: path_or_paths
    """A string or a dictionary with the source paths."""
    destination: path_or_paths
    """A string or a dictionary with the destination paths."""
    settings: dict[str, Any] | None = None
    """A dictionary with settings to pass to the pyspark job."""
    properties: dict[str, str] | None = None
    """A dictionary with the properties to pass to the pyspark job."""


class Pyspark(Task):
    """Task to run a pyspark job."""

    def __init__(self, spec: PysparkSpec, context: TaskContext) -> None:
        super().__init__(spec, context)
        self.spec: PysparkSpec
        self.pyspark = self.load_pyspark(spec.pyspark)

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

    @report
    def run(self) -> Self:
        self.srcs = make_absolute(self.spec.source, self.context.config)
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

        srcs = list(self.srcs.values()) if isinstance(self.srcs, dict) else self.srcs
        dsts = list(self.dsts.values()) if isinstance(self.dsts, dict) else self.dsts
        self.artifacts = [Artifact(source=srcs, destination=dsts)]

        return self
