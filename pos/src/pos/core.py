"""POS entry point: an otter runner with the scratchpad derived from release_uri."""

import asyncio
import errno
import os

from loguru import logger
from otter import Runner
from otter.manifest.model import Result

from pos.run import for_product, parse_release_uri

# cwd-relative, like the other config paths; the image's WORKDIR is the package root
CONFIG_PATH = 'config/config.yaml'


def main() -> None:
    os.environ.setdefault('POS_CONFIG_PATH', CONFIG_PATH)
    runner = Runner('pos')
    try:
        run = parse_release_uri(runner.config.release_uri or '')
        runner.specs = for_product(runner.specs, run.product, runner.config.step)
    except ValueError as e:
        logger.critical(e)
        raise SystemExit(errno.EINVAL)
    for key, value in run.scratchpad().items():
        runner.scratchpad.store(key, value)
    logger.info(f'run {run}')
    runner.start()
    runner.register_tasks('pos.tasks')
    s = asyncio.run(runner.run())
    if s.manifest.result not in [Result.PENDING, Result.SUCCESS]:
        logger.error(f'step {s.name} failed')
        raise SystemExit(1)
