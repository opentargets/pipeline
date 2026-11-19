# Sync bucket

import subprocess

from loguru import logger
from otter.task.model import Spec, Task, TaskContext
from otter.task.task_reporter import report
from otter.util.fs import check_dir

from pos.rclone.sync import strip_scheme

RCLONE_CONFIG = 'config/rclone.conf'


class SyncBucketError(Exception):
    """Base class for exceptions in this module."""


class SyncBucketSpec(Spec):
    """Configuration fields for the sync bucket task.

    This task has the following custom configuration fields:
        - source (str): The `gs://` URL of the directory to copy.
        - destination (Path): The path, relative to `work_path` to download the
            outputs to.
    """

    source: str
    destination: str


class SyncBucket(Task):
    def __init__(self, spec: SyncBucketSpec, context: TaskContext) -> None:
        super().__init__(spec, context)
        self.spec: SyncBucketSpec
        self.context: TaskContext

    @report
    def run(self) -> Task:
        if self.spec.destination.startswith('gs://'):
            destination_folder = self.spec.destination
        else:
            destination_folder = self.context.config.work_path.joinpath(self.spec.destination)
            logger.debug(f'checking if {destination_folder} exists. If not, create it')
            # Checking if the destination folder exists. If not, create it.
            check_dir(destination_folder)

        logger.debug(f'copying {self.spec.source} to {self.spec.destination}')
        pool = str(self.context.config.pool_size)
        # copy, never sync: nothing is deleted at the destination
        command = [
            'rclone',
            '--config',
            RCLONE_CONFIG,
            'copy',
            _remote(self.spec.source),
            _remote(str(destination_folder)),
            '--transfers',
            pool,
            '--checkers',
            pool,
        ]
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        return self


def _remote(path: str) -> str:
    return f'gcs:{strip_scheme(path)}' if path.startswith('gs://') else path
