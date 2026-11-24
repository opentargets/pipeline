"""Core module for the PPS application."""

import asyncio
from importlib import resources

from loguru import logger
from otter import Runner
from otter.manifest.model import Result


def task_packages() -> list[str]:
    """List the packages holding otter tasks.

    These are the generic ``pps.tasks`` and the ``tasks`` package of every datasource
    that defines one, e.g. ``pps.gwas_catalog.tasks``.
    """
    datasources = sorted(
        f'pps.{p.name}.tasks'
        for p in resources.files('pps').iterdir()
        if p.is_dir() and p.joinpath('tasks', '__init__.py').is_file()
    )
    return ['pps.tasks', *datasources]


def main() -> None:
    """Main entry point for the PPS application."""
    runner = Runner('pps')
    runner.start()
    for package in task_packages():
        runner.register_tasks(package)
    s = asyncio.run(runner.run())

    if s.manifest.result not in [Result.PENDING, Result.SUCCESS]:
        logger.error(f'step {s.name} failed')
        raise SystemExit(1)
