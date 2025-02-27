from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from typing import Self

from loguru import logger
from otter.manifest.model import Artifact
from otter.storage import get_remote_storage
from otter.task.model import Spec, Task, TaskContext
from otter.task.task_reporter import report
from otter.util.fs import check_destination

TRANSFORMER_PACKAGE = 'pts.transformers'


class TransformSpec(Spec):
    source: str | list[str]
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

        # sources
        source_list = spec.source if isinstance(spec.source, list) else [spec.source]
        srcs_local = [context.config.work_path / s for s in source_list]

        # if release_uri is not provided, check all files are available locally
        if not context.config.release_uri:
            for s in srcs_local:
                if not s.is_file():
                    raise FileNotFoundError(f'{s} not found locally and no release_uri provided')

        # build a map of local to remote sources
        self.l2r = {self.context.config.work_path / s: f'{context.config.release_uri}/{s}' for s in source_list}

        # destinations
        self.dst_local = context.config.work_path / spec.destination
        self.dst_remote = f'{context.config.release_uri}/{spec.destination}' if context.config.release_uri else None

        # transformation function to run
        self.transformer = self.load_transformer(spec.transformer)

    @staticmethod
    def load_transformer(transformer_name: str) -> Callable[[Path | list[Path], Path], None]:
        try:
            module = import_module(f'{TRANSFORMER_PACKAGE}.{transformer_name}')
            transformer: Callable[[Path | list[Path], Path], None] = getattr(module, transformer_name)
            if not callable(transformer):
                raise TypeError(f'{transformer_name} is not a callable')
            return transformer
        except ImportError:
            raise ModuleNotFoundError(f'{transformer_name} not found in {TRANSFORMER_PACKAGE}')

    @report
    def run(self) -> Self:
        # source list for the artifact metadata
        source = []

        # download the sources from remote storage if they are not present locally
        for src_local, src_remote in self.l2r.items():
            if src_local.is_file():
                source.append(str(src_local))
                logger.debug(f'using local source {src_local}')
            else:
                check_destination(src_local)
                remote_storage = get_remote_storage(src_remote)
                remote_storage.download_to_file(src_remote, src_local)
                source.append(src_remote)
                logger.debug(f'using remote source {src_remote} (downloaded to {src_local})')

        check_destination(self.dst_local, delete=True)

        # run the transformation
        s = list(self.l2r.keys())
        if len(s) == 1:
            s = s[0]
        self.transformer(s, self.dst_local)

        # upload the result to remote storage
        if self.dst_remote:
            remote_storage = get_remote_storage(self.dst_remote)
            remote_storage.upload(self.dst_local, self.dst_remote)
            logger.debug('upload successful')

        self.artifacts = [Artifact(source=source[0], destination=self.dst_remote or str(self.dst_local))]

        return self
