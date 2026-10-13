"""Core module for the PPS application."""

import asyncio

from loguru import logger
from otter import Runner
from otter.manifest.model import Result


def main() -> None:
    """Main entry point for the PPS application."""
    runner = Runner('pps')
    runner.start()
    runner.register_tasks('pps.tasks')
    s = asyncio.run(runner.run())

    if s.manifest.result not in [Result.PENDING, Result.SUCCESS]:
        logger.error(f'step {s.name} failed')
        raise SystemExit(1)
