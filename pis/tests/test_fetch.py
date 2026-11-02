"""`fetch_once` downloads a file once per date and records what it fetched."""

import asyncio
import hashlib
import json
import threading
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from loguru import logger
from otter.config.model import Config
from otter.manifest.model import Result
from otter.scratchpad.model import Scratchpad
from otter.task.model import TaskContext

from pis import fetch
from pis.fetch import SIDECAR_SUFFIX, FetchError, FetchRecord, fetch_once
from pis.tasks.fetch_once import FetchOnce, FetchOnceSpec

URL = 'https://example.org/files/data.txt'
BODY = b'some bytes\n'
DESTINATION = 'input/data.txt'


@pytest.fixture
def config(tmp_path: Path) -> Config:
    return Config(step='fetch', steps=['fetch'], release_uri=None, work_path=tmp_path)


@pytest.fixture(autouse=True)
def clock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fetch, '_now', lambda: datetime(2026, 11, 3, 8, 0, 0, tzinfo=UTC))


@pytest.fixture
def warnings() -> Iterator[list[str]]:
    messages: list[str] = []
    sink = logger.add(lambda m: messages.append(m.record['message']), level='WARNING')
    yield messages
    logger.remove(sink)


def _respond(monkeypatch: pytest.MonkeyPatch, status: int = 200, body: bytes = BODY, url: str = URL) -> None:
    def get(requested: str, timeout: int) -> SimpleNamespace:
        return SimpleNamespace(status_code=status, content=body, url=url, ok=200 <= status < 300)

    monkeypatch.setattr(fetch.requests, 'get', get)


async def _validate_task(task: FetchOnce) -> None:
    await task.validate()


def _validate(config: Config) -> FetchOnce:
    spec = FetchOnceSpec(name='fetch once', source=URL, destination=DESTINATION, as_of='2026-11-03')
    task = FetchOnce(spec, TaskContext(config, Scratchpad()))
    task.context.abort = threading.Event()
    task.start_run()
    task.finish_run()
    asyncio.run(_validate_task(task))
    return task


def _never_called(*args: object, **kwargs: object) -> None:
    raise AssertionError('requests.get must not be called')


def test_fetch_once_downloads_and_writes_sidecar(
    config: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _respond(monkeypatch, url='https://example.org/final/data.txt')

    record = fetch_once(URL, DESTINATION, config, '2026-11-03', extra={'period': 'P'})

    expected = {
        'source_url': 'https://example.org/final/data.txt',
        'fetched_at': '2026-11-03T08:00:00Z',
        'size': len(BODY),
        'sha256': hashlib.sha256(BODY).hexdigest(),
        'extra': {'period': 'P'},
    }
    assert (tmp_path / DESTINATION).read_bytes() == BODY
    assert json.loads((tmp_path / (DESTINATION + SIDECAR_SUFFIX)).read_text()) == expected
    assert record == FetchRecord(**expected)


def test_fetch_once_keeps_existing_file(config: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    destination = tmp_path / DESTINATION
    sidecar = tmp_path / (DESTINATION + SIDECAR_SUFFIX)
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b'kept')
    sidecar.write_text('{"kept": true}')
    monkeypatch.setattr(fetch.requests, 'get', _never_called)

    assert fetch_once(URL, DESTINATION, config, '2026-11-03') is None

    assert destination.read_bytes() == b'kept'
    assert sidecar.read_text() == '{"kept": true}'
    assert _validate(config).manifest.result == Result.SUCCESS


def test_validate_fails_on_missing_or_empty_destination(config: Config, tmp_path: Path) -> None:
    assert _validate(config).manifest.result == Result.FAILURE

    (tmp_path / DESTINATION).parent.mkdir(parents=True)
    (tmp_path / DESTINATION).write_bytes(b'')
    assert _validate(config).manifest.result == Result.FAILURE


def test_kept_file_without_sidecar_warns(
    config: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, warnings: list[str]
) -> None:
    destination = tmp_path / DESTINATION
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b'kept')
    monkeypatch.setattr(fetch.requests, 'get', _never_called)

    assert fetch_once(URL, DESTINATION, config, '2026-11-03') is None

    assert any(DESTINATION in m and 'provenance is unknown' in m for m in warnings)
    assert not (tmp_path / (DESTINATION + SIDECAR_SUFFIX)).exists()


@pytest.mark.parametrize(('status', 'body'), [(404, b'not found'), (200, b'')])
def test_fetch_once_fails_on_http_error_and_empty_body(
    config: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: int, body: bytes
) -> None:
    _respond(monkeypatch, status=status, body=body)

    with pytest.raises(FetchError, match=URL):
        fetch_once(URL, DESTINATION, config, '2026-11-03')

    assert not (tmp_path / DESTINATION).exists()
    assert not (tmp_path / (DESTINATION + SIDECAR_SUFFIX)).exists()


def test_fetch_once_warns_when_date_differs(
    config: Config, monkeypatch: pytest.MonkeyPatch, warnings: list[str]
) -> None:
    _respond(monkeypatch)

    assert fetch_once(URL, DESTINATION, config, '2026-11-01') is not None
    assert any('2026-11-01' in m and '2026-11-03' in m for m in warnings)

    warnings.clear()
    assert fetch_once(URL, 'input/other.txt', config, '2026-11-03') is not None
    assert not warnings
