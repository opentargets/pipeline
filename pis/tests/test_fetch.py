"""`fetch_once` downloads a file once per date and records what it fetched."""

import hashlib
import json
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest
from loguru import logger
from otter.config.model import Config
from otter.manifest.model import Result
from otter.scratchpad.model import Scratchpad
from otter.task.model import Task, TaskContext

from pis import fetch
from pis.fetch import SIDECAR_SUFFIX, FetchError, FetchRecord, fetch_once
from pis.tasks.fetch_once import FetchOnce, FetchOnceSpec

URL = 'https://example.org/files/data.txt'
BODY = b'some bytes\n'
DESTINATION = 'input/data.txt'


@pytest.fixture
def warnings() -> Iterator[list[str]]:
    messages: list[str] = []
    sink = logger.add(lambda m: messages.append(m.record['message']), level='WARNING')
    yield messages
    logger.remove(sink)


def _respond(
    monkeypatch: pytest.MonkeyPatch,
    status: int = 200,
    body: bytes = BODY,
    url: str = URL,
    headers: dict[str, str] | None = None,
) -> None:
    def get(requested: str, timeout: int) -> SimpleNamespace:
        return SimpleNamespace(status_code=status, content=body, url=url, ok=200 <= status < 300, headers=headers or {})

    monkeypatch.setattr(fetch.requests, 'get', get)


def _task(config: Config) -> FetchOnce:
    spec = FetchOnceSpec(name='fetch once', source=URL, destination=DESTINATION, as_of='2026-11-03')
    return FetchOnce(spec, TaskContext(config, Scratchpad()))


def _never_called(*args: object, **kwargs: object) -> None:
    raise AssertionError('requests.get must not be called')


def test_fetch_once_downloads_and_writes_sidecar(
    config: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _respond(
        monkeypatch,
        url='https://example.org/final/data.txt',
        headers={'Content-Type': 'text/plain', 'Last-Modified': 'Mon, 02 Nov 2026 10:00:00 GMT', 'ETag': '"abc"'},
    )

    record = fetch_once(URL, DESTINATION, config, '2026-11-03', extra={'period': 'P'})

    expected = {
        'source_url': 'https://example.org/final/data.txt',
        'requested_url': URL,
        'fetched_at': '2026-11-03T08:00:00Z',
        'size': len(BODY),
        'sha256': hashlib.sha256(BODY).hexdigest(),
        'content_type': 'text/plain',
        'last_modified': 'Mon, 02 Nov 2026 10:00:00 GMT',
        'etag': '"abc"',
        'extra': {'period': 'P'},
    }
    assert (tmp_path / DESTINATION).read_bytes() == BODY
    assert json.loads((tmp_path / (DESTINATION + SIDECAR_SUFFIX)).read_text()) == expected
    assert record == FetchRecord(**expected)


def test_sidecar_headers_are_null_when_absent(config: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _respond(monkeypatch)

    fetch_once(URL, DESTINATION, config, '2026-11-03')

    sidecar = json.loads((tmp_path / (DESTINATION + SIDECAR_SUFFIX)).read_text())
    assert sidecar['content_type'] is None
    assert sidecar['last_modified'] is None
    assert sidecar['etag'] is None


@pytest.mark.parametrize('content_type', ['text/html', 'Text/HTML; charset=UTF-8'])
def test_fetch_once_refuses_html(
    config: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, content_type: str
) -> None:
    _respond(monkeypatch, body=b'<html>error</html>', headers={'Content-Type': content_type})

    with pytest.raises(FetchError, match=f'{URL}.*{content_type}'):
        fetch_once(URL, DESTINATION, config, '2026-11-03')

    assert not (tmp_path / DESTINATION).exists()
    assert not (tmp_path / (DESTINATION + SIDECAR_SUFFIX)).exists()


def test_fetch_once_refuses_wrong_prefix(config: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _respond(monkeypatch, body=b'<!DOCTYPE html>' + b'x' * 100)

    with pytest.raises(FetchError) as error:
        fetch_once(URL, DESTINATION, config, '2026-11-03', expect=b'PK')

    assert URL in str(error.value)
    assert "b'PK'" in str(error.value)
    assert repr(b'<!DOCTYPE html>' + b'x' * 17) in str(error.value)
    assert not (tmp_path / DESTINATION).exists()
    assert not (tmp_path / (DESTINATION + SIDECAR_SUFFIX)).exists()


def test_fetch_once_accepts_expected_prefix(config: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _respond(monkeypatch, body=b'PK\x03\x04 zip')

    assert fetch_once(URL, DESTINATION, config, '2026-11-03', expect=b'PK') is not None

    assert (tmp_path / DESTINATION).read_bytes() == b'PK\x03\x04 zip'


def test_fetch_once_task_passes_expect(
    config: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, run: Callable[[Task], Task]
) -> None:
    _respond(monkeypatch)
    spec = FetchOnceSpec(name='fetch once', source=URL, destination=DESTINATION, as_of='2026-11-03', expect='PK')

    assert run(FetchOnce(spec, TaskContext(config, Scratchpad()))).manifest.result == Result.FAILURE
    assert not (tmp_path / DESTINATION).exists()


def test_fetch_once_keeps_existing_file(
    config: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, validate: Callable[[Task], Task]
) -> None:
    destination = tmp_path / DESTINATION
    sidecar = tmp_path / (DESTINATION + SIDECAR_SUFFIX)
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b'kept')
    sidecar.write_text('{"kept": true}')
    monkeypatch.setattr(fetch.requests, 'get', _never_called)

    assert fetch_once(URL, DESTINATION, config, '2026-11-03') is None

    assert destination.read_bytes() == b'kept'
    assert sidecar.read_text() == '{"kept": true}'
    assert validate(_task(config)).manifest.result == Result.SUCCESS


def test_validate_fails_on_missing_or_empty_destination(
    config: Config, tmp_path: Path, validate: Callable[[Task], Task]
) -> None:
    assert validate(_task(config)).manifest.result == Result.FAILURE

    (tmp_path / DESTINATION).parent.mkdir(parents=True)
    (tmp_path / DESTINATION).write_bytes(b'')
    assert validate(_task(config)).manifest.result == Result.FAILURE


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
