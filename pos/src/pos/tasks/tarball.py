# tarball task

import os
import subprocess
import tempfile
from contextlib import ExitStack
from pathlib import Path

from loguru import logger
from otter.manifest.model import Artifact
from otter.task.model import Spec, Task, TaskContext
from otter.task.task_reporter import report
from otter.util.errors import OtterError
from otter.util.fs import check_destination

from pos.rclone.sync import RCLONE_CONFIG, remote, tail


class TarballError(OtterError):
    """Base class for exceptions in this module."""


class TarballSpec(Spec):
    """Configuration fields for the create tarball task.

    source: The path of the folder to archive
    destination: The path of the tarball to create
    threads: pigz threads; 0 for one per CPU
    """

    source: Path
    destination: Path
    threads: int = 0


class Tarball(Task):
    def __init__(self, spec: TarballSpec, context: TaskContext) -> None:
        super().__init__(spec, context)
        self.spec: TarballSpec
        self.source: Path = spec.source
        self.local_path: Path = context.config.work_path / spec.destination
        self.remote_uri: str | None = None
        if self.context.config.release_uri:
            self.remote_uri = f'{self.context.config.release_uri}/{self.spec.destination}'
        self.destination = self.remote_uri or self.local_path

    @report
    def run(self) -> Task:
        if not self.source.exists():
            raise TarballError(f'{self.source} does not exist')
        if not self.remote_uri:
            check_destination(self.local_path, delete=True)
        threads = self.spec.threads or os.cpu_count() or 1
        stream_tarball(self.source, str(self.destination), threads)
        self.artifacts = [Artifact(source=str(self.source), destination=str(self.destination))]
        return self


def tarball_commands(source: Path | str, destination: str, threads: int) -> list[list[str]]:
    """The tar, pigz and rclone rcat stages: the archive is never whole on disk or in memory."""
    return [
        ['tar', '-C', str(source), '-cf', '-', '.'],
        ['pigz', '-p', str(threads)],
        ['rclone', '--config', RCLONE_CONFIG, 'rcat', remote(destination)],
    ]


def stream_tarball(source: Path | str, destination: str, threads: int) -> None:
    """Archive `source` to `destination`; raise if any pipe stage exits non-zero."""
    logger.debug(f'archiving {source} to {destination}')
    run_pipeline(tarball_commands(source, destination, threads))
    logger.info(f'archived {source} to {destination}')


def run_pipeline(commands: list[list[str]]) -> None:
    """Run commands as a shell pipe would, failing on any stage (pipefail)."""
    procs: list[subprocess.Popen] = []
    with ExitStack() as stack:
        errors = [stack.enter_context(tempfile.TemporaryFile()) for _ in commands]
        try:
            for i, command in enumerate(commands):
                stdin = procs[-1].stdout if procs else subprocess.DEVNULL
                stdout = subprocess.DEVNULL if i == len(commands) - 1 else subprocess.PIPE
                procs.append(subprocess.Popen(command, stdin=stdin, stdout=stdout, stderr=errors[i]))
                if i:
                    # only the next stage holds the read end, so a dead reader stops the writer
                    procs[-2].stdout.close()  # ty: ignore[unresolved-attribute]
            codes = [p.wait() for p in procs]
        finally:
            for p in procs:
                if p.poll() is None:
                    p.kill()
                    p.wait()
        failed = []
        for command, code, err in zip(commands, codes, errors, strict=True):
            if code != 0:
                err.seek(0)
                failed.append(f'{command[0]} exited {code}:\n{tail(err.read().decode(errors="replace"))}')
        if failed:
            raise TarballError('pipe failed; ' + '\n'.join(failed))
