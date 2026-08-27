"""Tests for the Cloud Logging rate-limit handling in the GCE sensor."""

from contextlib import contextmanager
from typing import Any
from unittest.mock import patch

import pytest
from google.api_core.exceptions import ResourceExhausted

from orchestration.operators.gce import (
    LOGGING_REQUEST_MAX_INTERVAL,
    RateLimitedLoggingClient,
    _backoff,
)


class _Log:
    """A logger that keeps nothing.

    A MagicMock would record every call, and a retry loop can make a great many of
    them before it gives up.
    """

    def __init__(self) -> None:
        self.warnings = 0

    def warning(self, *args: Any, **kwargs: Any) -> None:
        self.warnings += 1


@contextmanager
def _fake_clock():
    """Run retry loops on a simulated clock that only moves when sleep is called.

    The retry gives up on a wall-clock deadline. Patching sleep alone leaves that
    deadline hours away in real time while the loop spins at full speed, so the clock
    has to be patched with it or the test never finishes.
    """
    now = [0.0]

    def fake_sleep(seconds: float) -> None:
        now[0] += seconds

    with (
        patch('orchestration.operators.gce.time.sleep', side_effect=fake_sleep),
        patch('orchestration.operators.gce.time.monotonic', side_effect=lambda: now[0]),
    ):
        yield now


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
    """Fake a lazy pager: entries arrive page by page as the caller iterates.

    `fail_on` names the 0-based pages that raise ResourceExhausted the first time they
    are reached, mimicking the quota being hit part-way through a log. A generator
    cannot resume after raising, so each page is served from its own iterator.
    """
    seen: set[int] = set()

    def gen():
        for i, page in enumerate(pages):
            if i in fail_on and i not in seen:
                seen.add(i)
                raise ResourceExhausted('quota exceeded')
            yield from page

    class _Pager:
        def __init__(self) -> None:
            self._it = gen()

        def __iter__(self):
            return self

        def __next__(self):
            try:
                return next(self._it)
            except ResourceExhausted:
                # the real pager retries the failed page request; a fresh generator
                # replays from the start of the pages it has not yielded yet
                self._it = gen()
                raise

    return _Pager()


def test_a_rate_limited_later_page_is_retried() -> None:
    """The quota is hit while paging, not on the first request.

    The previous implementation wrapped only the call that builds the pager, so it
    guarded page one and nothing else -- and the caller pages through the whole log,
    which is where the quota is actually reached.
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

    assert 'e' in got, 'entries after the rate-limited page were lost'
    assert log.warnings >= 1, 'a rate-limited page should log and back off'


def test_entries_are_yielded_when_nothing_is_rate_limited() -> None:
    """The ordinary path returns every entry and never backs off."""
    client, log = _client()

    with (
        _fake_clock(),
        patch(
            'google.cloud.logging_v2.Client.list_entries',
            return_value=_pager([['a'], ['b', 'c']], fail_on=set()),
        ),
    ):
        got = list(client.list_entries(filter_='x'))

    assert got == ['a', 'b', 'c']
    assert log.warnings == 0


def test_retrying_gives_up_rather_than_waiting_forever() -> None:
    """A quota that never recovers must surface, not hang the task.

    The previous loop had no bound, so an unrecovering quota would have retried
    indefinitely -- indistinguishable from a hung task, and much harder to diagnose.
    """
    client, _ = _client()
    calls = {'n': 0}

    def always_limited() -> None:
        calls['n'] += 1
        raise ResourceExhausted('quota exceeded')

    with _fake_clock():
        with pytest.raises(ResourceExhausted):
            client._retrying(always_limited)

    assert calls['n'] > 1, 'should retry at least once before giving up'
    assert calls['n'] < 100, f'gave up only after {calls["n"]} attempts, which is not a bound'


def test_backoff_is_capped() -> None:
    """Waits grow but stay bounded, so one retry cannot sleep for hours."""
    interval = 2.0
    for _ in range(40):
        interval = _backoff(interval)
    assert interval <= LOGGING_REQUEST_MAX_INTERVAL
