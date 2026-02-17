"""Task that applies a transformation function to input files."""

from collections.abc import Callable
from importlib import import_module
from typing import Self

from loguru import logger
from otter.manifest.model import Artifact
from otter.storage.util import make_absolute
from otter.task.model import Spec, Task, TaskContext
from otter.task.task_reporter import report
from otter.util.fs import check_destination

TRANSFORMER_PACKAGE = 'pts.transformers'

path_or_paths = str | dict[str, str]
transformer_type = Callable[[str | dict[str, str], str | dict[str, str]], None]


class TransformSpec(Spec):
    """Configuration for the Transform task."""

    source: path_or_paths
    """A string or a dictionary with the source paths."""
    destination: path_or_paths
    """A string or a dictionary with the destination paths."""
    transformer: str
    """A string with the name of a transformer function.

        The function should be available in the package `pts.transformers`.

        It takes two arguments: the source path and the destination path.

        The function should read the data from the source path, transform it,
        and write the result to the destination path. Both paths will be local.
    """


class Transform(Task):
    """Task that applies a transformation function to input files."""

    def __init__(self, spec: TransformSpec, context: TaskContext) -> None:
        super().__init__(spec, context)
        self.spec: TransformSpec
        self.transformer = self.load_transformer(spec.transformer)

    def _prepare_dirs(self, paths: str | dict[str, str]) -> None:
        if isinstance(paths, dict):
            for path in paths.values():
                check_destination(path, delete=True)
        else:
            check_destination(paths, delete=True)

    @staticmethod
    def load_transformer(transformer_name: str) -> transformer_type:
        try:
            module = import_module(f'{TRANSFORMER_PACKAGE}.{transformer_name}')
            transformer: transformer_type = getattr(
                module,
                transformer_name,
            )
            if not callable(transformer):
                raise TypeError(f'{transformer_name} is not a callable')
            return transformer
        except ImportError:
            raise ModuleNotFoundError(f'{transformer_name} not found in {TRANSFORMER_PACKAGE}')

    @report
    def run(self) -> Self:
        self.srcs = make_absolute(self.spec.source, self.context.config)
        self.dsts = make_absolute(self.spec.destination, self.context.config)

        logger.debug(f'running transformer {self.spec.transformer}')
        logger.debug(f'with source {self.srcs} and destination {self.dsts}')

        # prepare the destination directories if running locally
        if not self.context.config.release_uri:
            self._prepare_dirs(self.dsts)

        self.transformer(self.srcs, self.dsts)

        srcs = list(self.srcs.values()) if isinstance(self.srcs, dict) else self.srcs
        dsts = list(self.dsts.values()) if isinstance(self.dsts, dict) else self.dsts
        self.artifacts = [Artifact(source=srcs, destination=dsts)]
        return self
