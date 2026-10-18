"""Log capture for the gene burden parser tests."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from loguru import logger


def logged(run: Callable[[], Any]) -> tuple[Any, str]:
    """`run`'s result and the messages it logged, concatenated."""
    messages: list[str] = []
    sink = logger.add(messages.append, format='{message}')
    try:
        result = run()
    finally:
        logger.remove(sink)
    return result, ''.join(messages)
