"""How long each stage of the variant step takes, logged when the stage ends."""

from __future__ import annotations

import functools
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager

from loguru import logger


@contextmanager
def timed(stage: str) -> Generator[None]:
    """Log how long the body took as `<stage> took <seconds>s`, once it returns."""
    start = time.perf_counter()
    yield
    logger.info('{} took {:.1f}s', stage, time.perf_counter() - start)


def timed_function[**P, T](stage: str, function: Callable[P, T]) -> Callable[P, T]:
    """`function`, each call of it inside `timed(stage)`."""

    @functools.wraps(function)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        with timed(stage):
            return function(*args, **kwargs)

    return wrapper
