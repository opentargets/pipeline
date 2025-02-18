import gzip
from pathlib import Path
from typing import Self

from loguru import logger
from otter.manifest.model import Artifact
from otter.storage import get_remote_storage
from otter.task.model import Spec, Task, TaskContext
from otter.task.task_reporter import report
from otter.util.fs import check_destination, check_source


class GzipSpec(Spec):
    source: Path
    destination: Path


class Gzip(Task):
    def __init__(self, spec: GzipSpec, context: TaskContext) -> None:
        super().__init__(spec, context)
        self.spec: GzipSpec
        self.source: Path = context.config.work_path / spec.source
        self.local_path: Path = context.config.work_path / spec.destination
        self.remote_uri: str | None = None
        if self.context.config.release_uri:
            self.remote_uri = f'{self.context.config.release_uri}/{self.spec.destination}'
        self.destination = self.remote_uri or self.local_path

    @report
    def run(self) -> Self:
        check_source(self.source)
        check_destination(self.local_path, delete=True)

        with gzip.open(self.local_path, 'wb') as gzip_file:
            gzip_file.write(self.source.read_bytes())

        logger.debug(f'gzipped {self.source} to {self.local_path}')

        # upload the result to remote storage
        if self.remote_uri:
            remote_storage = get_remote_storage(self.remote_uri)
            remote_storage.upload(self.local_path, self.remote_uri)
            logger.debug('upload successful')

        self.artifacts = [Artifact(source=str(self.source), destination=str(self.destination))]

        return self
