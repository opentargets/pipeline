from collections.abc import Callable
from importlib import import_module
from typing import Any, Self

from loguru import logger
from otter.task.model import Spec, Task, TaskContext
from otter.task.task_reporter import report

PYSPARK_PACKAGE = 'pts.pyspark'

type path_or_paths = str | dict[str, str]

type pyspark_entrypoint = Callable[
    [path_or_paths, path_or_paths, dict[str, Any] | None, dict[str, str] | None],
    None,
]


class PysparkSpec(Spec):
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
        src, dst = self.spec.source, self.spec.destination

        # build paths based on release_uri or work_path
        if prefix := self.context.config.release_uri or self.context.config.work_path:
            if isinstance(src, str):
                src = f'{prefix}/{src}'
            elif isinstance(src, dict):
                src = {k: f'{prefix}/{v}' for k, v in src.items()}
            if isinstance(dst, str):
                dst = f'{prefix}/{dst}'
            elif isinstance(dst, dict):
                dst = {k: f'{prefix}/{v}' for k, v in dst.items()}

        logger.debug(f'running pyspark job with spec: {self.spec}')

        # run the pyspark job
        self.pyspark(src, dst, self.spec.settings, self.spec.properties)
        logger.info('pyspark job completed successfully')

        # TODO: figure out artifacts
        self.artifacts = []

        return self
