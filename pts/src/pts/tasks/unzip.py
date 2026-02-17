"""Unzip a file and store locally."""

import zipfile
from pathlib import Path
from typing import Self

from loguru import logger
from otter.manifest.model import Artifact
from otter.storage.synchronous.handle import StorageHandle
from otter.task.model import Spec, Task, TaskContext
from otter.task.task_reporter import report
from otter.util.fs import check_destination


class UnzipSpec(Spec):
    """Specification for the unzip task."""

    source: str
    """The source URI of the file to unzip."""
    inner_file: str | None = None
    """A string with the name of the file inside the zip to extract. If it is not
        provided, it defaults to the name of the zip file without the extension."""
    destination: str
    """A string with the path to the destination. It will be appended to the
        :py:obj:`otter.config.model.Config.work_path`."""


class Unzip(Task):
    """Unzip a file and store locally."""

    def __init__(self, spec: UnzipSpec, context: TaskContext) -> None:
        super().__init__(spec, context)
        self.spec: UnzipSpec
        self.inner_file = spec.inner_file or str(Path(self.spec.source).stem)

    @report
    def run(self) -> Self:
        check_destination(self.spec.destination, delete=True)

        s = StorageHandle(self.spec.source, config=self.context.config)
        f = s.open('rb')
        logger.debug(f'unzipping {self.inner_file} from {self.spec.source} to {self.spec.destination}')

        with zipfile.ZipFile(f) as zip_file:
            if self.inner_file not in zip_file.namelist():
                raise FileNotFoundError(f'{self.inner_file} not found in {self.spec.source}')
            with zip_file.open(self.inner_file) as file:
                d = StorageHandle(self.spec.destination, config=self.context.config, force_local=True)
                with d.open('wb') as dest_file:
                    dest_file.write(file.read())
        logger.debug('unzip completed')

        self.artifacts = [Artifact(source=s.absolute, destination=d.absolute)]

        return self
