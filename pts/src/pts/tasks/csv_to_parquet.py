"""Task to convert a CSV file to Parquet format."""

from typing import Self

import polars as pl
from loguru import logger
from otter.manifest.model import Artifact
from otter.storage.synchronous.handle import StorageHandle
from otter.task.model import Spec, Task, TaskContext
from otter.task.task_reporter import report
from otter.util.fs import check_destination


class CsvToParquetSpec(Spec):
    """Specification for the CsvToParquet task."""

    source: str
    """The source URI of the CSV file, relative to the release root."""
    destination: str
    """The destination for the file, relative to the release root."""
    separator: str = ','
    """The separator used in the CSV file. Defaults to ','."""


class CsvToParquet(Task):
    """Task to convert a CSV file to Parquet format."""

    def __init__(self, spec: CsvToParquetSpec, context: TaskContext) -> None:
        super().__init__(spec, context)
        self.spec: CsvToParquetSpec

    @report
    def run(self) -> Self:
        check_destination(self.spec.destination, delete=True)

        s = StorageHandle(self.spec.source, config=self.context.config)
        d = StorageHandle(self.spec.destination, config=self.context.config)

        df = pl.read_csv(s.absolute, has_header=True, separator=self.spec.separator)
        df.write_parquet(d.absolute, compression='gzip')
        logger.info('transformation complete')

        self.artifacts = [Artifact(source=s.absolute, destination=d.absolute)]

        return self
