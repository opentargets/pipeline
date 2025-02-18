import zipfile
from pathlib import Path
from typing import Self

from loguru import logger
from otter.manifest.model import Artifact
from otter.storage import get_remote_storage
from otter.task.model import Spec, Task, TaskContext
from otter.task.task_reporter import report
from otter.util.fs import check_destination, check_source


class UnzipSpec(Spec):
    source: str
    inner_file: str | None = None
    destination: Path


class Unzip(Task):
    """Unzip a file and store locally (no upload)."""

    def __init__(self, spec: UnzipSpec, context: TaskContext) -> None:
        super().__init__(spec, context)
        self.spec: UnzipSpec

        self.src_local = context.config.work_path / spec.source
        self.src_remote: str | None = None
        if not self.src_local.is_file():
            if not self.context.config.release_uri:
                raise FileNotFoundError(f'{self.src_local} not found and no release uri provided')
            self.src_remote = f'{self.context.config.release_uri}/{spec.source}'

        self.source = self.src_remote or str(self.src_local)
        self.inner_file = spec.inner_file or Path(self.spec.source).stem
        self.destination = context.config.work_path / self.spec.destination

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

        check_destination(self.destination, delete=True)

        with zipfile.ZipFile(self.src_local) as zip_file:
            if self.inner_file not in zip_file.namelist():
                raise FileNotFoundError(f'{self.inner_file} not found in {self.spec.source}')
            with zip_file.open(self.inner_file) as file:
                Path(self.destination).write_bytes(file.read())
        logger.debug('unzip completed')

        self.artifacts = [Artifact(source=str(self.source), destination=str(self.destination))]

        return self
