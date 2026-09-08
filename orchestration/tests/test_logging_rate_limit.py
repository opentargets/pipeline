"""Tests for the Cloud Logging rate-limit handling in the GCE sensor."""

from contextlib import contextmanager
from typing import Any
from unittest.mock import patch

import pytest
from google.api_core.exceptions import ResourceExhausted
from google.cloud.logging_v2.services.logging_service_v2.pagers import ListLogEntriesPager
from google.cloud.logging_v2.types import ListLogEntriesRequest, ListLogEntriesResponse, LogEntry

from orchestration.operators.gce import LOGGING_REQUEST_INTERVAL, LOGGING_RETRY_MAX_WAIT, RateLimitedLoggingClient


class _Log:
    """A logger that keeps nothing.

    A MagicMock records every call, and a retry loop can make a great many of them.
    """

    def __init__(self) -> None:
        self.warnings = 0

    def warning(self, *args: Any, **kwargs: Any) -> None:
        self.warnings += 1


@contextmanager
def _fake_clock():
    """Run retry loops on a clock that only moves when sleep is called.

    The retry gives up on a wall-clock deadline, so patching sleep alone leaves that
    deadline minutes away in real time while the loop spins at full speed.
    """
    now = [0.0]

    def fake_sleep(seconds: float) -> None:
        now[0] += seconds

    with (
        patch('orchestration.operators.gce.time.sleep', side_effect=fake_sleep),
        patch('orchestration.operators.gce.time.monotonic', side_effect=lambda: now[0]),
    ):
        yield now


def _client(api) -> tuple[RateLimitedLoggingClient, _Log]:
    """A client with its base __init__ bypassed, so no credentials are needed.

    `api` stands in for the generated `list_log_entries`, the one call the client makes.
    The log stub comes back alongside the client: `client.log` is declared as a Logger,
    so reading the counter through it does not typecheck.
    """
    client = object.__new__(RateLimitedLoggingClient)
    log = _Log()
    client.log = log
    client.project = 'a-project'
    gapic = type('_Gapic', (), {'list_log_entries': staticmethod(api)})()
    client._logging_api = type('_Api', (), {'_gapic_api': gapic})()
    return client, log


def _pager(response: ListLogEntriesResponse) -> ListLogEntriesPager:
    """Wrap a page the way the generated client does, so the tests drive the real object.

    Production reads `entries` and `next_page_token` straight off this, through the
    pager's attribute delegation. A plain response would not exercise that.
    """
    return ListLogEntriesPager(
        method=lambda *a, **k: pytest.fail('the pager must never fetch a page itself'),
        request=ListLogEntriesRequest(resource_names=['projects/a-project']),
        response=response,
    )


def _entry(message: str) -> LogEntry:
    entry = LogEntry(log_name='projects/a-project/logs/l')
    entry.json_payload = {'message': message}  # ty:ignore[invalid-assignment]
    return entry


def _messages(client: RateLimitedLoggingClient) -> list[str]:
    return [entry.json_payload['message'] for entry in client.list_entries(filter_='x')]  # ty:ignore[invalid-return-type]


def _paged_api(pages: list[list[str]], fail_on: set[int]):
    """Serve real ListLogEntriesResponse pages, keyed by page token.

    `fail_on` names the 0-based pages that raise ResourceExhausted the first time they
    are requested. Nothing here remembers where the caller had got to: the page token
    on the request is the only position, exactly as it is against the real service.

    Comes back with the list of page tokens requested, so a test can assert that a
    rate-limited page was asked for again rather than skipped.
    """
    tokens = ['' if i == 0 else f'tok{i}' for i in range(len(pages))]
    by_token = {
        token: ListLogEntriesResponse(
            entries=[_entry(m) for m in page],
            next_page_token=tokens[i + 1] if i + 1 < len(pages) else '',
        )
        for i, (token, page) in enumerate(zip(tokens, pages, strict=True))
    }
    failed: set[int] = set()
    calls: list[str] = []

    def api(request):
        calls.append(request.page_token)
        page_number = tokens.index(request.page_token)
        if page_number in fail_on and page_number not in failed:
            failed.add(page_number)
            raise ResourceExhausted('quota exceeded')
        return _pager(by_token[request.page_token])

    return api, calls


def test_a_rate_limited_later_page_is_retried() -> None:
    """The quota is hit while paging, not on the first request.

    The caller pages through a whole step's log, which is where the quota is actually
    reached. A retry wrapped around the base client's generator cannot help: a
    generator that raises is closed, so the retry sees StopIteration and hands back a
    truncated log with no error at all. Only re-requesting the page by its token
    resumes the read, which is why the client pages explicitly.
    """
    api, requested = _paged_api([['a', 'b'], ['c', 'd'], ['e']], fail_on={1})
    client, log = _client(api)

    with _fake_clock():
        got = _messages(client)

    assert got == ['a', 'b', 'c', 'd', 'e']
    assert log.warnings == 1, 'the rate-limited page should back off exactly once'
    assert requested == ['', 'tok1', 'tok1', 'tok2'], 'the failed page should be re-requested by its token'


def test_the_first_page_is_retried_too() -> None:
    """Page one goes through the same budget as every other page."""
    api, requested = _paged_api([['a'], ['b']], fail_on={0})
    client, log = _client(api)

    with _fake_clock():
        got = _messages(client)

    assert got == ['a', 'b']
    assert log.warnings == 1
    assert requested == ['', '', 'tok1']


def test_an_unrecovering_quota_gives_up_rather_than_waiting_forever() -> None:
    """A quota that never recovers must surface, not hang the task.

    An unbounded retry is indistinguishable from a hung task, and much harder to
    diagnose than the error itself.
    """
    calls = {'n': 0}
    ceiling = 100

    def always_limited(request):
        calls['n'] += 1
        if calls['n'] > ceiling:
            # Without the deadline the loop never exits, and the test would hang rather
            # than fail. Bail out here so an unbounded retry is a red test, not a wait.
            raise AssertionError(f'still retrying after {ceiling} attempts; the retry is unbounded')
        raise ResourceExhausted('quota exceeded')

    client, _ = _client(always_limited)

    with _fake_clock(), pytest.raises(ResourceExhausted):
        _messages(client)

    assert calls['n'] > 1, 'should retry at least once before giving up'


def test_one_retry_budget_covers_the_whole_read() -> None:
    """The backoff and the deadline belong to the read, not to each request.

    A budget per request would let a log of n pages wait n * LOGGING_RETRY_MAX_WAIT
    before surfacing, and would drop the interval back to the minimum on every page, so
    the backoff would never grow however long the quota stayed exhausted.
    """
    pages = [['a'], ['b'], ['c'], ['d']]
    tokens = ['', 'tok1', 'tok2', 'tok3']
    served: set[str] = set()

    def api(request):
        # every page is rate-limited once, so the budget is charged on each of them
        if request.page_token not in served:
            served.add(request.page_token)
            raise ResourceExhausted('quota exceeded')
        i = tokens.index(request.page_token)
        return _pager(
            ListLogEntriesResponse(
                entries=[_entry(pages[i][0])],
                next_page_token=tokens[i + 1] if i + 1 < len(tokens) else '',
            )
        )

    client, _ = _client(api)

    with _fake_clock() as now:
        got = _messages(client)

    assert got == ['a', 'b', 'c', 'd']
    assert now[0] <= LOGGING_RETRY_MAX_WAIT, 'the whole read must fit in one retry budget'
    assert now[0] > LOGGING_REQUEST_INTERVAL * len(pages), 'the interval must keep growing across pages'
