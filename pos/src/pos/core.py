"""POS entry point: an otter runner with the scratchpad derived from release_uri."""

import asyncio
import errno
import os

from loguru import logger
from otter import Runner
from otter.manifest.model import Result
from otter.task.task_registry import TaskRegistry

from pos.run import OPTIONAL_STEPS, SCRATCH_PRODUCTS, for_mode, for_product, parse_release_uri, scratch_enabled

# cwd-relative, like the other config paths; the image's WORKDIR is the package root
CONFIG_PATH = 'config/config.yaml'


def setup() -> Runner:
    """A runner for the step, its specs, config and scratchpad set for the run and mode."""
    os.environ.setdefault('POS_CONFIG_PATH', CONFIG_PATH)
    runner = Runner('pos')
    try:
        scratch = scratch_enabled()
        run = parse_release_uri(runner.config.release_uri or '')
        # mode first: a scratch-only step refuses normal mode for every product
        runner.specs = for_mode(runner.specs, scratch, runner.config.step)
        runner.specs = for_product(runner.specs, run.product, runner.config.step)
    except ValueError as e:
        logger.critical(e)
        raise SystemExit(errno.EINVAL)
    if scratch:
        # otter's Config is frozen; the manifest goes under release_uri
        runner.config = runner.config.model_copy(update={'release_uri': run.scratch_release_uri()})
        runner.task_registry = TaskRegistry(runner.config, runner.scratchpad)
        runner.task_registry.register('otter.tasks')
        swapped = ', '.join(f'{k}={v}' for k, v in SCRATCH_PRODUCTS[run.product].items())
        logger.warning(f'SCRATCH MODE: release_uri={run.scratch_release_uri()}, {swapped}')
    for key, value in run.scratchpad(scratch).items():
        runner.scratchpad.store(key, value)
    runner.scratchpad.store('run_location', (runner.config.release_uri or '').rstrip('/'))
    logger.info(f'run {run}')
    runner.start()
    runner.register_tasks('pos.tasks')
    return runner


def main() -> None:
    runner = setup()
    step = runner.config.step
    if not runner.specs and step in OPTIONAL_STEPS:
        logger.info(f'step {step} does not apply to {runner.scratchpad.sentinel_dict["product"]}: nothing to run')
        return
    s = asyncio.run(runner.run())
    if s.manifest.result not in [Result.PENDING, Result.SUCCESS]:
        logger.error(f'step {s.name} failed')
        raise SystemExit(1)
