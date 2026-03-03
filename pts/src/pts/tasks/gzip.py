"""Task that gzips a file."""

import gzip
from pathlib import Path
from typing import Self

from loguru import logger
from otter.manifest.model import Artifact
from otter.storage.synchronous.handle import StorageHandle
from otter.task.model import Spec, Task, TaskContext
from otter.task.task_reporter import report
from otter.util.fs import check_source


class GzipSpec(Spec):
    """Specification for the gzip task."""

    source: Path
    """A string with the path, relative to work_dir, for the file to gzip. It will
        be appended with the :py:obj:`otter.config.model.Config.work_path`."""

    destination: str
    """The destination for the file, relative to the release root."""


class Gzip(Task):
    """Task that gzips a file.

    This will always use a local file as source (in the machine's filesystem), because
    we should never gzip a file from the outside without first making a copy of it
    with PIS.
    """

    def __init__(self, spec: GzipSpec, context: TaskContext) -> None:
        super().__init__(spec, context)
        self.spec: GzipSpec

    @report
    def run(self) -> Self:

        check_source(self.context.config.work_path / self.spec.source)

        s = StorageHandle(self.spec.source, config=self.context.config, force_local=True)
        sf = s.open('rb')

        d = StorageHandle(self.spec.destination, config=self.context.config)
        df = d.open('wb')
        with gzip.open(df, mode='wb') as gzip_file:
            gzip_file.write(sf.read())

        logger.debug(f'gzipped {s.absolute} to {d.absolute}')

        self.artifacts = [Artifact(source=str(s.absolute), destination=d.absolute)]

        return self
