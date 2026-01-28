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

path_or_paths = str | dict[str, str]
transformer_type = Callable[[Path | dict[str, Path], Path | dict[str, Path]], None]


class TransformSpec(Spec):
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
    def __init__(self, spec: TransformSpec, context: TaskContext) -> None:
        super().__init__(spec, context)
        self.spec: TransformSpec

        source_dict = spec.source if isinstance(spec.source, dict) else {'default': spec.source}
        self.srcs_local: dict[str, Path] = {k: context.config.work_path / v for k, v in source_dict.items()}
        self.l2r = {context.config.work_path / v: f'{context.config.release_uri}/{v}' for v in source_dict.values()}

        # if release_uri is not provided, check all files are available locally
        if not context.config.release_uri:
            for s in self.srcs_local.values():
                if not s.is_file():
                    raise FileNotFoundError(f'{s} not found locally and no release_uri provided')

        destination_dict = spec.destination if isinstance(spec.destination, dict) else {'default': spec.destination}
        self.dsts_local: dict[str, Path] = {k: context.config.work_path / v for k, v in destination_dict.items()}
        self.dsts_remote = {}
        if context.config.release_uri:
            self.dsts_remote = {k: f'{context.config.release_uri}/{v}' for k, v in destination_dict.items()}

        # transformation function to run
        self.transformer = self.load_transformer(spec.transformer)

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

        # check destinations
        for dst_local in self.dsts_local.values():
            check_destination(dst_local, delete=True)

        # run the transformation - pass dict or single Path based on whether input was dict
        s = self.srcs_local if isinstance(self.spec.source, dict) else self.srcs_local['default']
        d = self.dsts_local if isinstance(self.spec.destination, dict) else self.dsts_local['default']
        self.transformer(s, d)

        # upload the results to remote storage
        if self.dsts_remote:
            for key, dst_local in self.dsts_local.items():
                dst_remote = self.dsts_remote[key]
                remote_storage = get_remote_storage(dst_remote)
                remote_storage.upload(dst_local, dst_remote)
                logger.debug(f'upload successful: {dst_remote}')

        # build artifacts list
        self.artifacts = [
            Artifact(source=source[0], destination=self.dsts_remote.get(k) or str(v))
            for k, v in self.dsts_local.items()
        ]

        return self
