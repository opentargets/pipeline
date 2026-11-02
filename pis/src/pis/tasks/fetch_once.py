"""Downloads a file unless the destination already has it."""

from typing import Self

from otter.task.model import Spec, Task, TaskContext
from otter.task.task_reporter import report

from pis.fetch import fetch_once, validate_destination


class FetchOnceSpec(Spec):
    """Configuration fields for the fetch once task.

    This task has the following custom configuration fields:
        - source (str): The URL to download.
        - destination (str): The path, relative to `release_uri`, to write to.
        - as_of (str): The staging date, `YYYY-MM-DD`.
        - expect (str, optional): The prefix the body must start with.
    """

    source: str
    destination: str
    as_of: str
    expect: str | None = None


class FetchOnce(Task):
    """Download a file once per staging date.

    If the destination already exists it is kept and nothing is downloaded. A
    sidecar file beside the destination records the URL, time, size and sha256
    of the download, and the headers of the response.

    .. note:: `destination` will be prepended with either
        :py:obj:`otter.config.model.Config.release_uri` or
        :py:obj:`otter.config.model.Config.work_path`.
    """

    def __init__(self, spec: FetchOnceSpec, context: TaskContext) -> None:
        super().__init__(spec, context)
        self.spec: FetchOnceSpec

    @report
    def run(self) -> Self:
        expect = self.spec.expect.encode() if self.spec.expect is not None else None
        fetch_once(self.spec.source, self.spec.destination, self.context.config, self.spec.as_of, expect=expect)
        return self

    @report
    def validate(self) -> Self:
        """Check that the destination exists and is not empty."""
        validate_destination(self.spec.destination, self.context.config)
        return self
