import subprocess
import time
from pathlib import Path
from subprocess import CompletedProcess

from loguru import logger
from otter.util.errors import OtterError

RCLONE_CONFIG = 'config/rclone.conf'
# a NOTICE stats line every minute and one at the end, without -v's line per file
STATS_FLAGS = ['--stats', '1m', '--stats-one-line', '--stats-log-level', 'NOTICE']
TAIL_LINES = 50


class RcloneError(OtterError):
    """rclone exited non-zero."""


def strip_scheme(uri: str) -> str:
    """Turn `gcs:gs://b/x` into the rclone form `gcs:b/x`."""
    return uri.replace('gs://', '').replace('s3://', '')


def remote(path: str) -> str:
    """A `gs://` URI as an rclone `gcs:` remote, an `s3://` URI as `aws:`; anything else unchanged."""
    for scheme, name in (('gs://', 'gcs'), ('s3://', 'aws')):
        if path.startswith(scheme):
            return f'{name}:{path.removeprefix(scheme)}'
    return path


def tail(text: str, lines: int = TAIL_LINES) -> str:
    return '\n'.join(text.rstrip().splitlines()[-lines:])


def run_rclone(command: list[str], label: str) -> CompletedProcess:
    """Run rclone with its output captured: raise with its stderr tail on failure, log a summary line."""
    logger.debug(f'running: {" ".join(command)}')
    start = time.monotonic()
    rclone = subprocess.run(command, capture_output=True, text=True, check=False)
    elapsed = time.monotonic() - start
    if rclone.returncode != 0:
        raise RcloneError(f'rclone {label} exited {rclone.returncode}; stderr tail:\n{tail(rclone.stderr)}')
    last = tail(rclone.stderr, 1)
    logger.info(f'rclone {label} done in {elapsed:.0f} s' + (f': {last}' if last else ''))
    return rclone


class Sync:
    """A class to handle rclone sync operations."""

    def __init__(self, source: Path, destination: Path, config: Path, include_file: Path) -> None:
        self.source = source
        self.destination = destination
        self.config = config
        self.include_paths = include_file

    def sync(self, sync_flags: list[str]) -> CompletedProcess:
        """Syncs the source directory to the destination using rclone."""
        logger.info(f'Syncing {self.source} to {self.destination}')
        command = [
            'rclone',
            '--config',
            str(self.config),
            'sync',
            str(self.source),
            str(self.destination),
            *STATS_FLAGS,
            *sync_flags,
            '--include-from',
            str(self.include_paths),
        ]
        return run_rclone(command, f'sync {self.source} {self.destination}')
