"""Downloads a file unless the destination already has it."""

from typing import Self

from loguru import logger
from otter.storage.synchronous.handle import StorageHandle
from otter.task.model import Spec, Task, TaskContext
from otter.task.task_reporter import report
from otter.util.errors import TaskValidationError

from pis.fetch import fetch_once


class FetchOnceSpec(Spec):
    """Configuration fields for the fetch once task.

    This task has the following custom configuration fields:
        - source (str): The URL to download.
        - destination (str): The path, relative to `release_uri`, to write to.
        - as_of (str): The staging date, `YYYY-MM-DD`.
    """

    source: str
    destination: str
    as_of: str


class FetchOnce(Task):
    """Download a file once per staging date.

    If the destination already exists it is kept and nothing is downloaded. A
    sidecar file beside the destination records the URL, time, size and sha256
    of the download.

    .. note:: `destination` will be prepended with either
        :py:obj:`otter.config.model.Config.release_uri` or
        :py:obj:`otter.config.model.Config.work_path`.
    """

    def __init__(self, spec: FetchOnceSpec, context: TaskContext) -> None:
        super().__init__(spec, context)
        self.spec: FetchOnceSpec

    @report
    def run(self) -> Self:
        fetch_once(self.spec.source, self.spec.destination, self.context.config, self.spec.as_of)
        return self

    @report
    def validate(self) -> Self:
        """Check that the destination exists and is not empty."""
        try:
            size = StorageHandle(self.spec.destination, config=self.context.config).stat().size
        except Exception as e:
            raise TaskValidationError(f'{self.spec.destination} does not exist') from e
        if not size:
            logger.error(f'{self.spec.destination} is empty')
            raise TaskValidationError(f'{self.spec.destination} is empty')
        return self
