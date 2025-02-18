from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from typing import Self

from loguru import logger
from otter.manifest.model import Artifact
from otter.storage import get_remote_storage
from otter.task.model import Spec, Task, TaskContext
from otter.task.task_reporter import report
from otter.util.fs import check_destination, check_source

TRANSFORMER_PACKAGE = 'pts.transformers'


class TransformSpec(Spec):
    source: str
    destination: str
    transformer: str
    """A string with the name of a transformer function.

        The function should be available in the package `pts.transformers`.

        It takes two arguments: the source path and the destination path.

        The function should read the data from the source path, transform it,
        and write the result to the destination path. Both paths will be local.
    """


class Transform(Task):
    def __init__(self, spec: TransformSpec, context: TaskContext) -> None:
        super().__init__(spec, context)
        self.spec: TransformSpec

        self.src_local = context.config.work_path / spec.source
        self.src_remote: str | None = None
        if not self.src_local.is_file():
            if not self.context.config.release_uri:
                raise FileNotFoundError(f'{self.src_local} not found and no release uri provided')
            self.src_remote = f'{self.context.config.release_uri}/{spec.source}'

        self.dst_local: Path = context.config.work_path / spec.destination
        self.dst_remote: str | None = None
        if self.context.config.release_uri:
            self.dst_remote = f'{self.context.config.release_uri}/{self.spec.destination}'

        self.source = self.src_remote or str(self.src_local)
        self.destination = self.dst_remote or str(self.dst_local)

        self.transformer: Callable[[Path, Path], None] = self.load_transformer(spec.transformer)

    @staticmethod
    def load_transformer(transformer_name: str) -> Callable[[Path, Path], None]:
        try:
            module = import_module(f'{TRANSFORMER_PACKAGE}.{transformer_name}')
            transformer: Callable[[Path, Path], None] = getattr(module, transformer_name)
            if not callable(transformer):
                raise TypeError(f'{transformer_name} is not callable')
            return transformer
        except ImportError:
            raise ModuleNotFoundError(f'{transformer_name} not found in {TRANSFORMER_PACKAGE}')

    @report
    def run(self) -> Self:
        # download the source from remote storage
        if self.src_remote:
            check_destination(self.src_local)
            remote_storage = get_remote_storage(self.src_remote)
            remote_storage.download_to_file(self.src_remote, self.src_local)
            logger.debug(f'downloaded {self.src_remote} to {self.src_local}')
        else:
            check_source(self.src_local)

        check_destination(self.dst_local, delete=True)

        # run the transformation
        self.transformer(self.src_local, self.dst_local)

        # upload the result to remote storage
        if self.dst_remote:
            remote_storage = get_remote_storage(self.dst_remote)
            remote_storage.upload(self.dst_local, self.dst_remote)
            logger.debug('upload successful')

        self.artifacts = [Artifact(source=self.source, destination=self.destination)]

        return self
