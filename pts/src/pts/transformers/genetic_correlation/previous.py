"""The previous run of the genetic correlation DAG, which a run reuses what has not changed from.

A task is given the previous run's output folder, and reads the previous version of each of its
own datasets under the same name: the previous `munged` of `<run>/output/munged` is
`<previous output>/munged`.
"""

from __future__ import annotations

from typing import Annotated

from otter.storage.synchronous.handle import StorageHandle
from otter.util.errors import NotFoundError
from pydantic import BeforeValidator

PreviousOutput = Annotated[str | None, BeforeValidator(lambda value: value or None)]
"""The previous run's output folder; None, or an empty string, for a run with nothing to reuse."""


def previous_path(previous_output: str | None, current: str) -> str | None:
    """The previous run's version of the dataset at `current`, or None without a previous run.

    Raises:
        FileNotFoundError: if there is a previous run but it has no such dataset.
    """
    if previous_output is None:
        return None
    path = f'{previous_output.rstrip("/")}/{current.rstrip("/").rsplit("/", 1)[-1]}'
    try:
        StorageHandle(path).stat()
    except NotFoundError:
        raise FileNotFoundError(f'the previous run has no {path}') from None
    return path
