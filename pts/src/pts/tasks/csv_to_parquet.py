from pathlib import Path
from typing import Self

import polars as pl
from loguru import logger
from otter.manifest.model import Artifact
from otter.storage import get_remote_storage
from otter.task.model import Spec, Task, TaskContext
from otter.task.task_reporter import report
from otter.util.fs import check_destination, check_source


class CsvToParquetSpec(Spec):
    source: str
    destination: str
    separator: str = ','


class CsvToParquet(Task):
    def __init__(self, spec: CsvToParquetSpec, context: TaskContext) -> None:
        super().__init__(spec, context)
        self.spec: CsvToParquetSpec

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

        df = pl.read_csv(self.src_local, has_header=True, separator=self.spec.separator)
        df.write_parquet(self.dst_local, compression='gzip')
        logger.info('transformation complete')

        # upload the result to remote storage
        if self.dst_remote:
            remote_storage = get_remote_storage(self.dst_remote)
            remote_storage.upload(self.dst_local, self.dst_remote)
            logger.debug('upload successful')

        self.artifacts = [Artifact(source=self.source, destination=self.destination)]

        return self
