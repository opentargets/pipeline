# Copy one file between buckets with rclone

from otter.task.model import Spec, Task, TaskContext
from otter.task.task_reporter import report

from pos.rclone.sync import RCLONE_CONFIG, STATS_FLAGS, remote, run_rclone


class RcloneCopytoSpec(Spec):
    """Configuration fields for the rclone copyto task.

    source: the `gs://` or `s3://` URI of the file to copy
    destination: the `gs://` or `s3://` URI to copy it to
    """

    source: str
    destination: str


class RcloneCopyto(Task):
    def __init__(self, spec: RcloneCopytoSpec, context: TaskContext) -> None:
        super().__init__(spec, context)
        self.spec: RcloneCopytoSpec

    @report
    def run(self) -> Task:
        source, destination = remote(self.spec.source), remote(self.spec.destination)
        command = ['rclone', '--config', RCLONE_CONFIG, 'copyto', source, destination, *STATS_FLAGS]
        run_rclone(command, f'copyto {self.spec.source} {self.spec.destination}')
        return self
