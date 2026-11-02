"""`fetch_pmda` finds the consolidated New Drugs PDF on the PMDA page and fetches it once."""

import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest
from otter.config.model import Config
from otter.manifest.model import Result
from otter.scratchpad.model import Scratchpad
from otter.task.model import Task, TaskContext

from pis import fetch
from pis.fetch import SIDECAR_SUFFIX, FetchError
from pis.tasks.fetch_pmda import FetchPmda, FetchPmdaSpec, PmdaLink, find_new_drugs_pdf

FIXTURES = Path(__file__).parent / 'fixtures' / 'pmda'
PAGE_URL = 'https://www.pmda.go.jp/english/review-services/reviews/approved-information/drugs/0002.html'
PDF_URL = 'https://www.pmda.go.jp/files/000281190.pdf'
PDF_BODY = b'%PDF-1.7 new drugs\n'
DESTINATION = 'input/pmda.pdf'


def _page(name: str) -> str:
    return (FIXTURES / name).read_text()


def _task(config: Config) -> FetchPmda:
    spec = FetchPmdaSpec(name='fetch pmda', page=PAGE_URL, destination=DESTINATION, as_of='2026-11-03')
    return FetchPmda(spec, TaskContext(config, Scratchpad()))


def _respond(monkeypatch: pytest.MonkeyPatch, page: str) -> list[str]:
    requested: list[str] = []

    def get(url: str, timeout: int) -> SimpleNamespace:
        requested.append(url)
        if url == PAGE_URL:
            body = page.encode()
        elif url == PDF_URL:
            body = PDF_BODY
        else:
            raise AssertionError(f'unexpected request for {url}')
        return SimpleNamespace(status_code=200, content=body, text=page, url=url, ok=True)

    monkeypatch.setattr(fetch.requests, 'get', get)
    return requested


def test_finds_todays_pdf() -> None:
    link = find_new_drugs_pdf(_page('page_2026-11.html'), PAGE_URL)

    assert link == PmdaLink(PDF_URL, 'April 2004 to February 2026 [3.06 MB]')


def test_older_layout_picks_all() -> None:
    link = find_new_drugs_pdf(_page('page_2025-01.html'), PAGE_URL)

    assert link.url == 'https://www.pmda.go.jp/files/000272313.pdf'
    assert link.text.startswith('All')


@pytest.mark.parametrize(
    ('name', 'urls'),
    [
        ('page_none.html', ['/files/000281191.pdf', '/files/000281194.pdf']),
        ('page_two.html', ['/files/000281190.pdf', '/files/000281193.pdf']),
    ],
)
def test_no_match_and_two_matches_fail(name: str, urls: list[str]) -> None:
    with pytest.raises(FetchError) as error:
        find_new_drugs_pdf(_page(name), PAGE_URL)

    assert PAGE_URL in str(error.value)
    for url in urls:
        assert f'https://www.pmda.go.jp{url}' in str(error.value)
    assert '000281192' not in str(error.value)


def test_fetch_pmda_records_period(
    config: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, run: Callable[[Task], Task]
) -> None:
    requested = _respond(monkeypatch, _page('page_2026-11.html'))

    assert run(_task(config)).manifest.result != Result.FAILURE

    assert requested == [PAGE_URL, PDF_URL]
    assert (tmp_path / DESTINATION).read_bytes() == PDF_BODY
    sidecar = json.loads((tmp_path / (DESTINATION + SIDECAR_SUFFIX)).read_text())
    assert sidecar['extra'] == {'period': 'April 2004 to February 2026', 'page': PAGE_URL}
    assert sidecar['source_url'] == PDF_URL


def test_fetch_pmda_keeps_existing_without_fetching_page(
    config: Config,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    run: Callable[[Task], Task],
    validate: Callable[[Task], Task],
) -> None:
    (tmp_path / DESTINATION).parent.mkdir(parents=True)
    (tmp_path / DESTINATION).write_bytes(b'kept')

    def get(*args: object, **kwargs: object) -> None:
        raise AssertionError('requests.get must not be called')

    monkeypatch.setattr(fetch.requests, 'get', get)

    assert run(_task(config)).manifest.result != Result.FAILURE
    assert validate(_task(config)).manifest.result == Result.SUCCESS
    assert (tmp_path / DESTINATION).read_bytes() == b'kept'


def test_validate_fails_on_missing_destination(config: Config, validate: Callable[[Task], Task]) -> None:
    assert validate(_task(config)).manifest.result == Result.FAILURE
