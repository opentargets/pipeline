"""Fixtures shared by the fetch task tests."""

import asyncio
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pytest
from otter.config.model import Config
from otter.task.model import Task

from pis import fetch


@pytest.fixture
def config(tmp_path: Path) -> Config:
    return Config(step='fetch', steps=['fetch'], release_uri=None, work_path=tmp_path)


@pytest.fixture(autouse=True)
def clock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fetch, '_now', lambda: datetime(2026, 11, 3, 8, 0, 0, tzinfo=UTC))


@pytest.fixture
def validate() -> Callable[[Task], Task]:
    """Run a task's `validate` through its reporter, which records the outcome in the manifest."""

    async def _validate(task: Task) -> None:
        await task.validate()

    def run(task: Task) -> Task:
        task.context.abort = threading.Event()
        task.start_run()
        task.finish_run()
        asyncio.run(_validate(task))
        return task

    return run


@pytest.fixture
def run() -> Callable[[Task], Task]:
    """Run a task's `run` through its reporter, which records the outcome in the manifest."""

    async def _run(task: Task) -> None:
        await task.run()

    def execute(task: Task) -> Task:
        task.context.abort = threading.Event()
        asyncio.run(_run(task))
        return task

    return execute
