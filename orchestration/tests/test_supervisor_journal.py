"""Tests for the supervisor's append-only journal."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from orchestration.supervisor.journal import Journal, JournalEvent


def _event(event_type: str = 'step_completed', step: str = 'pts_target', try_number: int = 1) -> JournalEvent:
    return JournalEvent(
        event_type=event_type,
        step=step,
        try_number=try_number,
        at=datetime(2026, 7, 21, 14, 0, tzinfo=UTC),
        payload={'duration': 3600.0},
    )


class FakeBucket:
    """Stands in for a GCS bucket, storing objects in a dict."""

    def __init__(self) -> None:
        self.objects: dict[str, str] = {}

    def blob(self, name: str) -> FakeBlob:
        return FakeBlob(self, name)

    def list_blobs(self, prefix: str) -> list[FakeBlob]:
        return [FakeBlob(self, n) for n in sorted(self.objects) if n.startswith(prefix)]


class FakeBlob:
    def __init__(self, bucket: FakeBucket, name: str) -> None:
        self.bucket = bucket
        self.name = name

    def exists(self) -> bool:
        return self.name in self.bucket.objects

    def upload_from_string(self, data: str, content_type: str = '') -> None:
        self.bucket.objects[self.name] = data

    def download_as_text(self) -> str:
        return self.bucket.objects[self.name]


def _journal() -> tuple[Journal, FakeBucket]:
    bucket = FakeBucket()
    return Journal(bucket=bucket, prefix='ds/run/_agent/journal'), bucket


class TestJournalEventKey:
    def test_key_combines_type_step_and_try(self) -> None:
        assert _event().key == 'step_completed-pts_target-1'

    def test_the_same_event_twice_has_the_same_key(self) -> None:
        assert _event().key == _event().key

    def test_a_later_try_is_a_different_event(self) -> None:
        assert _event(try_number=1).key != _event(try_number=2).key

    def test_a_step_is_optional(self) -> None:
        """Run-level events such as run_finished have no step."""
        e = JournalEvent(event_type='run_finished', at=datetime(2026, 7, 21, tzinfo=UTC))
        assert e.key == 'run_finished'


class TestJournalAppend:
    def test_append_writes_one_object_per_event(self) -> None:
        journal, bucket = _journal()
        journal.append(_event())
        assert len(bucket.objects) == 1
        assert next(iter(bucket.objects)).startswith('ds/run/_agent/journal/')

    def test_append_is_idempotent(self) -> None:
        """The agent re-derives state every wakeup, so it will re-observe the same completions.

        A second append of the same key must not add an entry.
        """
        journal, bucket = _journal()
        assert journal.append(_event()) is True
        assert journal.append(_event()) is False
        assert len(bucket.objects) == 1

    def test_distinct_events_both_land(self) -> None:
        journal, bucket = _journal()
        journal.append(_event(step='pts_target'))
        journal.append(_event(step='pts_disease'))
        assert len(bucket.objects) == 2

    def test_the_object_name_carries_the_key(self) -> None:
        journal, bucket = _journal()
        journal.append(_event())
        assert 'step_completed-pts_target-1' in next(iter(bucket.objects))


class TestJournalEventValidation:
    """A `/` in a key component would let one key become a path-prefix of another."""

    def test_event_type_cannot_contain_a_slash(self) -> None:
        with pytest.raises(ValidationError):
            JournalEvent(event_type='a/b', at=datetime(2026, 7, 21, tzinfo=UTC))

    def test_step_cannot_contain_a_slash(self) -> None:
        with pytest.raises(ValidationError):
            JournalEvent(event_type='step_completed', step='pts/target', at=datetime(2026, 7, 21, tzinfo=UTC))


class TestJournalKeyCollisions:
    """A key must be an exact match, never a coincidental suffix of a different one."""

    def test_pts_target_and_pts_pre_target_do_not_collide(self) -> None:
        """The two real steps this repo actually has, ruling out the concrete case."""
        journal, _ = _journal()
        journal.append(_event(step='pts_pre_target'))
        assert journal.has(_event(step='pts_target').key) is False

    def test_hyphenated_keys_do_not_collide(self) -> None:
        """A string-suffix check would treat `a-b` as recorded once `x-a-b` had been.

        `-x-a-b.json` ends with `-a-b.json`. Real step names use underscores, so this
        pathological case needs a synthetic hyphenated key to surface here — it pins the
        object layout that makes a key an exact path segment instead.
        """
        journal, _ = _journal()
        journal.append(JournalEvent(event_type='x-a-b', at=datetime(2026, 7, 21, 14, 0, tzinfo=UTC)))
        assert journal.has('a-b') is False


class TestJournalRead:
    def test_read_returns_what_was_appended(self) -> None:
        journal, _ = _journal()
        journal.append(_event(step='pts_target'))
        journal.append(_event(step='pts_disease'))
        assert {e.step for e in journal.read()} == {'pts_target', 'pts_disease'}

    def test_read_round_trips_the_payload(self) -> None:
        journal, _ = _journal()
        journal.append(_event())
        assert journal.read()[0].payload == {'duration': 3600.0}

    def test_an_empty_journal_reads_as_empty(self) -> None:
        journal, _ = _journal()
        assert journal.read() == []

    def test_has_reports_membership_without_reading_everything(self) -> None:
        journal, _ = _journal()
        journal.append(_event())
        assert journal.has(_event().key) is True
        assert journal.has('step_failed-pts_target-1') is False

    def test_read_orders_by_time_even_when_key_order_disagrees(self) -> None:
        """Object names now sort by key first, not by timestamp.

        Chronological order has to come from an explicit sort rather than from listing
        order.
        """
        journal, _ = _journal()
        early_but_alphabetically_last = JournalEvent(
            event_type='step_completed', step='zzz_step', try_number=1, at=datetime(2026, 7, 21, 10, 0, tzinfo=UTC)
        )
        later_but_alphabetically_first = JournalEvent(
            event_type='step_completed', step='aaa_step', try_number=1, at=datetime(2026, 7, 21, 14, 0, tzinfo=UTC)
        )
        journal.append(early_but_alphabetically_last)
        journal.append(later_but_alphabetically_first)
        assert [e.step for e in journal.read()] == ['zzz_step', 'aaa_step']

    def test_read_does_not_sweep_in_a_sibling_prefix(self) -> None:
        """`ds/run/_agent/journal` must not read events belonging to `.../journal2`.

        An unanchored `str.startswith` prefix check would let a sibling run's or a
        differently-suffixed journal's objects bleed into this one's history.
        """
        journal, bucket = _journal()
        journal.append(_event(step='pts_target'))
        sibling = Journal(bucket=bucket, prefix='ds/run/_agent/journal2')
        sibling.append(_event(step='pts_disease'))
        assert {e.step for e in journal.read()} == {'pts_target'}
