"""Tests for the supervisor's append-only journal."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from orchestration.supervisor.journal import Journal, JournalEvent, heartbeat_event, is_heartbeat


def _event(
    event_type: str = 'step_completed',
    step: str = 'pts_target',
    try_number: int = 1,
    map_index: int | None = None,
) -> JournalEvent:
    return JournalEvent(
        event_type=event_type,
        step=step,
        try_number=try_number,
        map_index=map_index,
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
    def test_key_combines_type_step_and_tagged_try(self) -> None:
        assert _event().key == 'step_completed-pts_target-t1'

    def test_the_same_event_twice_has_the_same_key(self) -> None:
        assert _event().key == _event().key

    def test_a_later_try_is_a_different_event(self) -> None:
        assert _event(try_number=1).key != _event(try_number=2).key

    def test_a_step_is_optional(self) -> None:
        """Run-level events such as run_finished have no step."""
        e = JournalEvent(event_type='run_finished', at=datetime(2026, 7, 21, tzinfo=UTC))
        assert e.key == 'run_finished'

    def test_an_unmapped_task_keeps_the_key_unchanged(self) -> None:
        """-1 is Airflow's value for a task instance outside a mapped operator, and by far the common case.

        The key must not grow a segment for it.
        """
        assert _event(map_index=-1).key == 'step_completed-pts_target-t1'

    def test_no_map_index_keeps_the_key_unchanged(self) -> None:
        assert _event(map_index=None).key == 'step_completed-pts_target-t1'

    def test_a_mapped_instance_gets_a_tagged_key_segment(self) -> None:
        assert _event(map_index=2).key == 'step_completed-pts_target-t1-m2'

    def test_two_mapped_instances_of_the_same_step_get_different_keys(self) -> None:
        """The defect this closes: N shards sharing one task_id used to collapse onto one key.

        A `stall_detected` event for shard 1 would be recorded first, and `Journal.append`
        would then silently drop shard 3's event as a duplicate of it.
        """
        assert _event(map_index=0).key != _event(map_index=3).key

    def test_a_plain_unmapped_single_attempt_event_has_a_stable_readable_key(self) -> None:
        """The common case — one attempt, no mapped instance — reads plainly, with exactly one tag."""
        e = JournalEvent(event_type='step_failed', step='pts_target', try_number=1,
                         at=datetime(2026, 7, 21, tzinfo=UTC))
        assert e.key == 'step_failed-pts_target-t1'

    def test_try_number_one_and_map_index_one_are_tagged_so_they_cannot_be_confused(self) -> None:
        """The ambiguity this format exists to close.

        A positional join collapses `try_number=1, map_index=-1/None` and
        `try_number=None, map_index=1` onto the identical trailing `-1` — one means a
        first attempt outside a mapped operator, the other means an unattempted
        (`try_number` unknown) mapped instance 1, and a plain join cannot tell them
        apart. Tagging each with its own letter makes them different strings by
        construction.
        """
        try_number_one = JournalEvent(event_type='step_failed', step='pts_target', try_number=1,
                                      at=datetime(2026, 7, 21, tzinfo=UTC))
        map_index_one = JournalEvent(event_type='step_failed', step='pts_target', map_index=1,
                                     at=datetime(2026, 7, 21, tzinfo=UTC))
        assert try_number_one.key != map_index_one.key
        assert try_number_one.key == 'step_failed-pts_target-t1'
        assert map_index_one.key == 'step_failed-pts_target-m1'

    def test_a_try_number_and_a_map_index_together_are_both_tagged_and_in_order(self) -> None:
        e = JournalEvent(event_type='stall_detected', step='pts_target', try_number=2, map_index=7,
                         at=datetime(2026, 7, 21, tzinfo=UTC))
        assert e.key == 'stall_detected-pts_target-t2-m7'


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
        assert 'step_completed-pts_target-t1' in next(iter(bucket.objects))


class TestJournalEventValidation:
    """A `/` in a key component would let one key become a path-prefix of another."""

    def test_event_type_cannot_contain_a_slash(self) -> None:
        with pytest.raises(ValidationError):
            JournalEvent(event_type='a/b', at=datetime(2026, 7, 21, tzinfo=UTC))

    def test_step_cannot_contain_a_slash(self) -> None:
        with pytest.raises(ValidationError):
            JournalEvent(event_type='step_completed', step='pts/target', at=datetime(2026, 7, 21, tzinfo=UTC))

    def test_step_cannot_be_a_qualified_task_id(self) -> None:
        """`stall.baseline_from_journal` explains why this mismatch fails silently and permanently."""
        with pytest.raises(ValidationError):
            JournalEvent(
                event_type='step_completed',
                step='pts_target.run_pts_target',
                at=datetime(2026, 7, 21, tzinfo=UTC),
            )

    def test_step_accepts_a_bare_step_name(self) -> None:
        e = JournalEvent(event_type='step_completed', step='pts_target', at=datetime(2026, 7, 21, tzinfo=UTC))
        assert e.step == 'pts_target'

    def test_step_accepts_none_for_a_run_level_event(self) -> None:
        e = JournalEvent(event_type='run_finished', step=None, at=datetime(2026, 7, 21, tzinfo=UTC))
        assert e.step is None


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


class TestHeartbeatEvent:
    def test_two_heartbeats_a_minute_apart_get_different_keys(self) -> None:
        """The defect this exists to prevent: a constant key would no-op after the first.

        `Journal.append` skips writing whenever `has(event.key)` is already true, so if
        every heartbeat shared one key, only the first wakeup's would ever be recorded.
        """
        first = heartbeat_event(datetime(2026, 7, 21, 14, 0, tzinfo=UTC))
        second = heartbeat_event(datetime(2026, 7, 21, 14, 10, tzinfo=UTC))
        assert first.key != second.key

    def test_the_same_wakeup_produces_the_same_key(self) -> None:
        at = datetime(2026, 7, 21, 14, 0, tzinfo=UTC)
        assert heartbeat_event(at).key == heartbeat_event(at).key

    def test_the_key_sorts_chronologically(self) -> None:
        """A raw journal listing sorts by key (object-name prefix) first, not by `at`.

        See `journal.py`'s `Journal.read` docstring: object names are not stored in
        timestamp order, so a key that does not itself sort chronologically would read
        as scrambled in a raw `gsutil ls`/`list_blobs` listing.
        """
        earlier = heartbeat_event(datetime(2026, 7, 21, 14, 0, tzinfo=UTC))
        later = heartbeat_event(datetime(2026, 7, 21, 14, 10, tzinfo=UTC))
        assert sorted([later.key, earlier.key]) == [earlier.key, later.key]

    def test_a_heartbeat_carries_no_step(self) -> None:
        """A heartbeat is a run-level fact, not about any one step."""
        assert heartbeat_event(datetime(2026, 7, 21, 14, 0, tzinfo=UTC)).step is None

    def test_a_heartbeat_is_recognised_as_one(self) -> None:
        assert is_heartbeat(heartbeat_event(datetime(2026, 7, 21, 14, 0, tzinfo=UTC))) is True

    def test_an_ordinary_event_is_not_a_heartbeat(self) -> None:
        assert is_heartbeat(_event()) is False

    def test_a_heartbeat_round_trips_through_the_journal(self) -> None:
        journal, _ = _journal()
        journal.append(heartbeat_event(datetime(2026, 7, 21, 14, 0, tzinfo=UTC)))
        journal.append(heartbeat_event(datetime(2026, 7, 21, 14, 10, tzinfo=UTC)))
        assert sum(1 for e in journal.read() if is_heartbeat(e)) == 2


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
        assert journal.has('step_failed-pts_target-t1') is False

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
