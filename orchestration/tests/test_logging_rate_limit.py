"""Tests for the Cloud Logging rate-limit handling in the GCE sensor."""

from contextlib import contextmanager
from typing import Any
from unittest.mock import patch

import pytest
from google.api_core.exceptions import ResourceExhausted

from orchestration.operators.gce import RateLimitedLoggingClient


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
        yield


def _client() -> tuple[RateLimitedLoggingClient, _Log]:
    """A client with its base __init__ bypassed, so no credentials are needed.

    The log stub comes back alongside it: `client.log` is declared as a Logger, so
    reading the counter through it does not typecheck.
    """
    client = object.__new__(RateLimitedLoggingClient)
    log = _Log()
    client.log = log
    return client, log


def _pager(pages: list[list[str]], fail_on: set[int]):
    """Fake a lazy pager: page one on construction, the rest as the caller iterates.

    `fail_on` names the 0-based pages that raise ResourceExhausted the first time they
    are reached. Position is tracked across a failure the way the real pager's page
    token is, so a retry resumes rather than replaying entries already yielded.
    """
    flat = [(page_number, entry) for page_number, page in enumerate(pages) for entry in page]

    class _Pager:
        def __init__(self) -> None:
            self.position = 0
            self.failed: set[int] = set()

        def __iter__(self) -> '_Pager':
            return self

        def __next__(self) -> str:
            if self.position >= len(flat):
                raise StopIteration
            page_number, entry = flat[self.position]
            if page_number in fail_on and page_number not in self.failed:
                self.failed.add(page_number)
                raise ResourceExhausted('quota exceeded')
            self.position += 1
            return entry

    return _Pager()


def test_a_rate_limited_later_page_is_retried() -> None:
    """The quota is hit while paging, not on the first request.

    The previous implementation wrapped only the call that builds the pager, so it
    guarded page one and nothing else -- and the caller pages through the whole log,
    which is where the quota is actually reached. Entries after the rate-limited page
    were lost and the step failed.
    """
    client, log = _client()

    with (
        _fake_clock(),
        patch(
            'google.cloud.logging_v2.Client.list_entries',
            return_value=_pager([['a', 'b'], ['c', 'd'], ['e']], fail_on={1}),
        ),
    ):
        got = list(client.list_entries(filter_='x'))

    assert got == ['a', 'b', 'c', 'd', 'e']
    assert log.warnings == 1, 'the rate-limited page should back off exactly once'


def test_retrying_gives_up_rather_than_waiting_forever() -> None:
    """A quota that never recovers must surface, not hang the task.

    The previous loop had no bound, so an unrecovering quota would have retried
    indefinitely -- indistinguishable from a hung task, and much harder to diagnose.
    """
    client, _ = _client()
    calls = {'n': 0}
    ceiling = 100

    def always_limited() -> None:
        calls['n'] += 1
        if calls['n'] > ceiling:
            # Without the deadline the loop never exits, and the test would hang rather
            # than fail. Bail out here so an unbounded retry is a red test, not a wait.
            raise AssertionError(f'still retrying after {ceiling} attempts; the retry is unbounded')
        raise ResourceExhausted('quota exceeded')

    with _fake_clock():
        with pytest.raises(ResourceExhausted):
            client._retrying(always_limited)

    assert calls['n'] > 1, 'should retry at least once before giving up'
