"""Tests for stall detection and snapshot assembly."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from orchestration.supervisor.airflow import DagRun, TaskInstance
from orchestration.supervisor.journal import JournalEvent, heartbeat_event
from orchestration.supervisor.snapshot import Snapshot, render_snapshot, take_snapshot
from orchestration.supervisor.stall import (
    _RUN_STALL_WAKEUP_THRESHOLD,
    RunStallVerdict,
    StallVerdict,
    _wakeups_since_step_event,
    baseline_from_journal,
    run_stalled,
    stalled,
)

NOW = datetime(2026, 7, 21, 20, 0, tzinfo=UTC)


def _running(task_id: str = 'pts_target.run_pts_target', minutes: int = 30) -> TaskInstance:
    return TaskInstance(task_id=task_id, state='running', start_date=NOW - timedelta(minutes=minutes))


def _completed(step: str, duration: float) -> JournalEvent:
    return JournalEvent(event_type='step_completed', step=step, try_number=1, at=NOW,
                        payload={'duration': duration})


def _client(tasks: list[TaskInstance], state: str = 'running') -> MagicMock:
    client = MagicMock()
    client.dag_run.return_value = DagRun(dag_run_id='r', state=state, start_date=NOW)
    client.task_instances.return_value = tasks
    return client


def _journal(events: list[JournalEvent] | None = None) -> MagicMock:
    journal = MagicMock()
    journal.read.return_value = events or []
    return journal


class TestBaselineFromJournal:
    def test_takes_the_observed_maximum_per_step(self) -> None:
        base = baseline_from_journal([_completed('a', 100.0), _completed('a', 300.0), _completed('a', 200.0)])
        assert base['a'] == 300.0

    def test_ignores_events_that_are_not_completions(self) -> None:
        events = [_completed('a', 100.0),
                  JournalEvent(event_type='stall_detected', step='a', at=NOW, payload={'duration': 9999.0})]
        assert baseline_from_journal(events)['a'] == 100.0

    def test_ignores_a_completion_with_no_duration(self) -> None:
        events = [JournalEvent(event_type='step_completed', step='a', try_number=1, at=NOW, payload={})]
        assert baseline_from_journal(events) == {}

    def test_an_empty_journal_gives_an_empty_baseline(self) -> None:
        assert baseline_from_journal([]) == {}

    def test_a_non_numeric_duration_does_not_destroy_other_steps_baselines(self) -> None:
        """One malformed event must degrade to a partial baseline, not raise and lose all of it."""
        events = [_completed('a', 100.0),
                  JournalEvent(event_type='step_completed', step='b', try_number=1, at=NOW,
                              payload={'duration': 'N/A'})]
        assert baseline_from_journal(events) == {'a': 100.0}

    def test_heartbeats_do_not_corrupt_the_baseline(self) -> None:
        """The claim `stall.py`'s module docstring makes about heartbeats, verified rather than assumed.

        A heartbeat's `event_type` is never `'step_completed'`, so it must be filtered
        out by that check alone — independent of whatever its payload happens to
        contain. A heartbeat never actually carries a `step` or a `duration`, so this
        pins a pathological one that does, with a `step` matching a real completion's,
        to exercise the `event_type` guard itself: if the rogue carried no `step`
        (`None`), the `event.step is None` clause would filter it out on its own and
        the test would stay green even with the `event_type` check deleted — the exact
        way this test used to pass for the wrong reason.
        """
        heartbeat = heartbeat_event(NOW)
        rogue = JournalEvent(event_type=heartbeat.event_type, step='a', at=NOW, payload={'duration': 999999.0})
        events = [_completed('a', 100.0), rogue]
        assert baseline_from_journal(events) == {'a': 100.0}


class TestStalled:
    def test_a_task_within_its_baseline_is_not_stalled(self) -> None:
        assert stalled(_running(minutes=30), {'pts_target': 3600.0}, NOW) is None

    def test_a_task_far_past_its_baseline_is_stalled(self) -> None:
        verdict = stalled(_running(minutes=300), {'pts_target': 3600.0}, NOW)
        assert verdict is not None
        assert verdict.basis == 'history'

    def test_a_sibling_task_in_the_group_does_not_inherit_the_run_tasks_baseline(self) -> None:
        """`step_from_task_id` collapses every task in a group onto the same step name.

        Without gating on `is_run_task`, a `delete_vm_` sibling would inherit the run
        task's history threshold instead of falling to the ceiling — see the module
        docstring for the concrete failure this produces in both directions.
        """
        sibling = TaskInstance(task_id='pts_target.delete_vm_pts_target', state='running',
                                start_date=NOW - timedelta(minutes=500))
        verdict = stalled(sibling, {'pts_target': 3600.0}, NOW)
        assert verdict is not None
        assert verdict.basis == 'ceiling'

    def test_a_task_with_no_history_falls_back_to_the_ceiling(self) -> None:
        """Most steps have no baseline on early runs, so this is the common path."""
        verdict = stalled(_running(minutes=500), {}, NOW)
        assert verdict is not None
        assert verdict.basis == 'ceiling'

    def test_a_task_with_no_history_under_the_ceiling_is_not_stalled(self) -> None:
        assert stalled(_running(minutes=30), {}, NOW) is None

    def test_only_running_tasks_are_judged(self) -> None:
        done = TaskInstance(task_id='t', state='success', start_date=NOW - timedelta(days=2))
        assert stalled(done, {}, NOW) is None

    def test_a_running_task_with_no_start_date_is_not_judged(self) -> None:
        """A task can be running before Airflow records a start date."""
        assert stalled(TaskInstance(task_id='t', state='running'), {}, NOW) is None

    def test_a_deferred_task_past_its_threshold_is_stalled(self) -> None:
        """Deferred has to be judged too, or the most common step type is invisible.

        Every PIS/PTS GCE step runs a deferrable sensor, so a hung VM parks the task
        instance in `deferred` rather than `running`.
        """
        deferred = TaskInstance(task_id='t', state='deferred', start_date=NOW - timedelta(minutes=500))
        verdict = stalled(deferred, {}, NOW)
        assert verdict is not None
        assert verdict.basis == 'ceiling'

    def test_a_restarting_task_past_its_threshold_is_stalled(self) -> None:
        """`restarting` has to be judged like `running`.

        It is non-terminal: the task was running and got externally interrupted
        (cleared, or its worker died) and still holds whatever cloud resource it had.
        No operator here implements `on_kill`, so nothing reclaims that resource on
        its own.
        """
        restarting = TaskInstance(task_id='t', state='restarting', start_date=NOW - timedelta(minutes=500))
        verdict = stalled(restarting, {}, NOW)
        assert verdict is not None
        assert verdict.basis == 'ceiling'

    def test_the_verdict_reports_elapsed_and_threshold(self) -> None:
        verdict = stalled(_running(minutes=500), {}, NOW)
        assert verdict is not None
        assert verdict.elapsed == 500 * 60
        assert verdict.threshold == 6 * 60 * 60

    def test_two_stalled_mapped_instances_of_the_same_task_get_distinct_verdicts(self) -> None:
        """N Google Batch shards share one task_id; only map_index tells their verdicts apart."""
        shard_a = TaskInstance(task_id='run_gentropy_variant_annotation', map_index=0, state='running',
                                start_date=NOW - timedelta(minutes=500))
        shard_b = TaskInstance(task_id='run_gentropy_variant_annotation', map_index=3, state='running',
                                start_date=NOW - timedelta(minutes=500))
        verdict_a = stalled(shard_a, {}, NOW)
        verdict_b = stalled(shard_b, {}, NOW)
        assert verdict_a is not None
        assert verdict_b is not None
        assert verdict_a.task_id != verdict_b.task_id
        assert verdict_a.task_id == 'run_gentropy_variant_annotation[0]'
        assert verdict_b.task_id == 'run_gentropy_variant_annotation[3]'

    def test_an_unmapped_stalled_task_reports_the_bare_task_id(self) -> None:
        """map_index -1 is the overwhelming majority; the verdict must not grow a suffix for it."""
        verdict = stalled(_running(minutes=500), {}, NOW)
        assert verdict is not None
        assert verdict.task_id == 'pts_target.run_pts_target'


def _heartbeats(n: int, start: datetime = NOW) -> list[JournalEvent]:
    """`n` consecutive heartbeats, a minute apart, starting at `start`."""
    return [heartbeat_event(start + timedelta(minutes=i)) for i in range(n)]


class TestWakeupsSinceStepEvent:
    def test_an_empty_journal_has_seen_zero_wakeups(self) -> None:
        assert _wakeups_since_step_event([]) == 0

    def test_counts_heartbeats_after_the_last_completion(self) -> None:
        events = [_completed('a', 100.0), *_heartbeats(3, start=NOW + timedelta(minutes=1))]
        assert _wakeups_since_step_event(events) == 3

    def test_a_completion_as_the_most_recent_event_gives_zero(self) -> None:
        """The stop condition itself: without it, this would count every prior heartbeat instead of stopping."""
        events = [*_heartbeats(3), _completed('a', 100.0)]
        assert _wakeups_since_step_event(events) == 0

    def test_a_failure_resets_the_count_as_well_as_a_completion(self) -> None:
        failure = JournalEvent(event_type='step_failed', step='a', try_number=1, at=NOW, payload={})
        events = [failure, *_heartbeats(2, start=NOW + timedelta(minutes=1))]
        assert _wakeups_since_step_event(events) == 2

    def test_a_journal_with_only_heartbeats_counts_them_all(self) -> None:
        assert _wakeups_since_step_event(_heartbeats(4)) == 4

    def test_non_heartbeat_non_step_events_are_skipped_without_resetting_or_advancing(self) -> None:
        """A `stall_detected`/`observation_started`/etc. in between must be invisible to this count."""
        events = [
            *_heartbeats(2),
            JournalEvent(event_type='stall_detected', step='a', at=NOW + timedelta(minutes=2), payload={}),
            *_heartbeats(2, start=NOW + timedelta(minutes=3)),
        ]
        assert _wakeups_since_step_event(events) == 4


class TestRunStalled:
    def test_a_run_that_is_not_running_never_stalls_this_way(self) -> None:
        """Neither signature applies once the run is no longer `running`."""
        for state in ('success', 'failed', None):
            assert run_stalled(state, {'pending': 5}, [], []) is None

    def test_stuck_trigger_fires_when_nothing_active_but_work_is_pending(self) -> None:
        verdict = run_stalled('running', {'pending': 3}, [], [])
        assert verdict is not None
        assert verdict.reason == 'stuck_trigger'
        assert verdict.pending == 3

    def test_stuck_trigger_does_not_fire_when_something_is_active(self) -> None:
        assert run_stalled('running', {'running': 1, 'pending': 3}, [], _heartbeats(10)) is None

    def test_stuck_trigger_does_not_fire_when_nothing_is_pending_either(self) -> None:
        """Zero active and zero pending is not this rule's shape — see `run_stalled`'s docstring.

        Deliberately paired with plenty of silent heartbeats too: with zero active tasks
        `len(stalls) < active` (`0 < 0`) is false, so a version of `run_stalled` that
        merely fell through past a narrower `active == 0 and pending > 0` check, instead
        of returning early on `active == 0` regardless of `pending`, would carry on to
        the `'no_progress'` branch and fire it with `active_tasks=0` — this pins that it
        does not.
        """
        assert run_stalled('running', {'success': 40}, [], _heartbeats(20)) is None

    def test_stuck_trigger_needs_no_journal_history_at_all(self) -> None:
        """Detectable from a single snapshot — an empty events list must still be enough."""
        verdict = run_stalled('running', {'pending': 1}, [], [])
        assert verdict is not None
        assert verdict.reason == 'stuck_trigger'

    def test_no_progress_does_not_fire_while_an_active_task_is_still_unflagged(self) -> None:
        """The legitimately-long-step case this must not trigger on.

        One active task, no `StallVerdict` for it yet (still within its own `stalled()`
        threshold), and plenty of silent wakeups: this must stay quiet, because the
        active task fully explains the silence and `stalled()` owns its fate.
        """
        verdict = run_stalled('running', {'running': 1}, [], _heartbeats(20))
        assert verdict is None

    def test_no_progress_fires_once_every_active_task_is_already_flagged(self) -> None:
        stalls = [StallVerdict(task_id='pts_target.run_pts_target', elapsed=99999.0, threshold=21600.0,
                               basis='ceiling')]
        verdict = run_stalled('running', {'running': 1}, stalls, _heartbeats(6), wakeup_threshold=6)
        assert verdict is not None
        assert verdict.reason == 'no_progress'
        assert verdict.wakeups == 6
        assert verdict.active_tasks == 1

    def test_no_progress_does_not_fire_below_the_wakeup_threshold(self) -> None:
        stalls = [StallVerdict(task_id='t', elapsed=99999.0, threshold=21600.0, basis='ceiling')]
        assert run_stalled('running', {'running': 1}, stalls, _heartbeats(5), wakeup_threshold=6) is None

    def test_no_progress_does_not_fire_at_the_threshold_minus_one(self) -> None:
        """Boundary check: exactly one short of the threshold must not be enough."""
        stalls = [StallVerdict(task_id='t', elapsed=99999.0, threshold=21600.0, basis='ceiling')]
        assert run_stalled('running', {'running': 1}, stalls, _heartbeats(5), wakeup_threshold=6) is None

    def test_no_progress_fires_exactly_at_the_threshold(self) -> None:
        stalls = [StallVerdict(task_id='t', elapsed=99999.0, threshold=21600.0, basis='ceiling')]
        verdict = run_stalled('running', {'running': 1}, stalls, _heartbeats(6), wakeup_threshold=6)
        assert verdict is not None

    def test_no_progress_does_not_fire_when_only_some_active_tasks_are_flagged(self) -> None:
        """Two active tasks, only one already stalled: the unflagged one still explains the silence."""
        stalls = [StallVerdict(task_id='t', elapsed=99999.0, threshold=21600.0, basis='ceiling')]
        assert run_stalled('running', {'running': 2}, stalls, _heartbeats(20)) is None

    def test_no_progress_fires_when_every_active_task_across_states_is_flagged(self) -> None:
        """`deferred` and `restarting` count as active too, exactly as `stalled()` judges them."""
        stalls = [
            StallVerdict(task_id='a', elapsed=99999.0, threshold=21600.0, basis='ceiling'),
            StallVerdict(task_id='b', elapsed=99999.0, threshold=21600.0, basis='ceiling'),
        ]
        counts = {'running': 1, 'deferred': 1}
        verdict = run_stalled('running', counts, stalls, _heartbeats(6), wakeup_threshold=6)
        assert verdict is not None
        assert verdict.active_tasks == 2

    def test_a_recent_step_completion_resets_the_silence(self) -> None:
        """A step_completed anywhere in the last `wakeup_threshold` wakeups clears the count."""
        stalls = [StallVerdict(task_id='t', elapsed=99999.0, threshold=21600.0, basis='ceiling')]
        events = [
            *_heartbeats(2),
            JournalEvent(event_type='step_completed', step='pts_disease', try_number=1,
                        at=NOW + timedelta(minutes=3), payload={'duration': 10.0}),
            *_heartbeats(2, start=NOW + timedelta(minutes=4)),
        ]
        assert run_stalled('running', {'running': 1}, stalls, events, wakeup_threshold=6) is None

    def test_a_step_failure_also_resets_the_silence(self) -> None:
        """`step_failed` is a step event too, not only `step_completed`."""
        stalls = [StallVerdict(task_id='t', elapsed=99999.0, threshold=21600.0, basis='ceiling')]
        events = [
            JournalEvent(event_type='step_failed', step='pts_disease', try_number=1, at=NOW, payload={}),
            *_heartbeats(6, start=NOW + timedelta(minutes=1)),
        ]
        wakeups_only = run_stalled('running', {'running': 1}, stalls, events, wakeup_threshold=6)
        assert wakeups_only is not None
        assert wakeups_only.wakeups == 6

    def test_stuck_trigger_does_not_fire_at_a_runs_healthy_opening_minutes(self) -> None:
        """A run's opening minutes must not read as stuck.

        The scheduler has created all 132 task instances as `pending` and moved a root
        task through `scheduled` into `queued`, but nothing has reached `running` yet.
        `active == 0` here, exactly like a truly stuck run, so this shape must be told
        apart on `queued` alone, not on `active`.
        """
        assert run_stalled('running', {'queued': 1, 'pending': 131}, [], []) is None

    def test_stuck_trigger_does_not_fire_during_a_scheduled_handoff(self) -> None:
        """A routine step hand-off must not read as stuck.

        A step finishing and the next one being handed off sits in `scheduled`, not
        yet `queued` or `running`. `active == 0` again, and again this is ordinary
        progress, not a stall.
        """
        counts = {'scheduled': 1, 'success': 40, 'pending': 91}
        assert run_stalled('running', counts, [], []) is None

    def test_stuck_trigger_does_not_fire_during_legitimate_retry_backoff(self) -> None:
        """Legitimate retry backoff must not read as stuck.

        `up_for_retry` is bounded to ~6 minutes across `stage_jar_*`'s three retries at
        a 2-minute delay (`unified_pipeline.py:299`) — legitimate backoff, not a
        scheduler that has given up.
        """
        counts = {'up_for_retry': 1, 'pending': 100}
        assert run_stalled('running', counts, [], []) is None

    def test_stuck_trigger_still_fires_when_nothing_is_moving_at_all(self) -> None:
        """The real signal must survive alongside the three false-positive fixes above.

        No active task, nothing queued, scheduled or retrying — only completed steps
        and tasks that have never been touched.
        """
        verdict = run_stalled('running', {'success': 32, 'pending': 100}, [], [])
        assert verdict is not None
        assert verdict.reason == 'stuck_trigger'
        assert verdict.pending == 100

    def test_stuck_trigger_and_no_progress_can_never_both_fire(self) -> None:
        """Mutually exclusive by construction: one requires zero active tasks, the other at least one."""
        active_verdict = run_stalled('running', {'running': 1, 'pending': 1},
                                     [StallVerdict(task_id='t', elapsed=1.0, threshold=1.0, basis='ceiling')],
                                     _heartbeats(6), wakeup_threshold=6)
        idle_verdict = run_stalled('running', {'pending': 1}, [], [])
        assert active_verdict is not None
        assert active_verdict.reason == 'no_progress'
        assert idle_verdict is not None
        assert idle_verdict.reason == 'stuck_trigger'


class TestTakeSnapshot:
    def test_counts_tasks_by_state(self) -> None:
        tasks = [
            TaskInstance(task_id='a', state='success'),
            TaskInstance(task_id='b', state='success'),
            TaskInstance(task_id='c', state='failed'),
            TaskInstance(task_id='d', state='running', start_date=NOW),
        ]
        snap = take_snapshot(_client(tasks), _journal(), 'unified_pipeline', 'r', NOW)
        assert snap.counts == {'success': 2, 'failed': 1, 'running': 1}

    def test_lists_failed_and_running_task_ids(self) -> None:
        tasks = [TaskInstance(task_id='c', state='failed'),
                 TaskInstance(task_id='d', state='running', start_date=NOW)]
        snap = take_snapshot(_client(tasks), _journal(), 'unified_pipeline', 'r', NOW)
        assert snap.failed == ['c']
        assert snap.running == ['d']

    def test_distinguishes_mapped_running_task_instances(self) -> None:
        """The two Google Batch steps expand N task instances under one shared task_id.

        Without map_index, `running` would list the same id N times with nothing to
        tell the shards apart.
        """
        tasks = [TaskInstance(task_id='run_gentropy_variant_annotation', map_index=0,
                               state='running', start_date=NOW),
                 TaskInstance(task_id='run_gentropy_variant_annotation', map_index=1,
                               state='running', start_date=NOW)]
        snap = take_snapshot(_client(tasks), _journal(), 'unified_pipeline', 'r', NOW)
        assert snap.running == ['run_gentropy_variant_annotation[0]', 'run_gentropy_variant_annotation[1]']

    def test_distinguishes_mapped_failed_task_instances(self) -> None:
        tasks = [TaskInstance(task_id='t', map_index=0, state='failed'),
                 TaskInstance(task_id='t', map_index=1, state='failed')]
        snap = take_snapshot(_client(tasks), _journal(), 'unified_pipeline', 'r', NOW)
        assert snap.failed == ['t[0]', 't[1]']

    def test_a_task_with_no_state_is_counted_as_pending(self) -> None:
        """Airflow reports a null state for a task instance not yet scheduled."""
        snap = take_snapshot(_client([TaskInstance(task_id='a')]), _journal(), 'unified_pipeline', 'r', NOW)
        assert snap.counts == {'pending': 1}

    def test_stalls_are_detected_from_the_journal_baseline(self) -> None:
        """A step_completed event for the task gives it a baseline, so this exercises the history path.

        The journal carries the bare step name (`pts_target`), the same spelling
        `usage.StepUsage.step` and the GCP `step` label already use; `stalled` converts
        the group-qualified Airflow task id down to it before looking up the baseline.
        """
        tasks = [TaskInstance(task_id='pts_target.run_pts_target', state='running',
                               start_date=NOW - timedelta(hours=9))]
        journal = _journal([_completed('pts_target', 3600.0)])
        snap = take_snapshot(_client(tasks), journal, 'unified_pipeline', 'r', NOW)
        assert [s.task_id for s in snap.stalls] == ['pts_target.run_pts_target']
        assert snap.stalls[0].basis == 'history'

    def test_a_qualified_task_id_cannot_reach_the_journal_at_all(self) -> None:
        """The failure mode this used to pin: a journal keyed on the qualified task id, not the bare step.

        `pts_target` (the bare step name `stalled` looks baselines up under) and
        `pts_target.run_pts_target` (the fully-qualified Airflow task id) are different
        strings, and a journal entry written under the qualified id would silently and
        permanently fall back to the ceiling instead of raising — indistinguishable from
        an honest first run. `JournalEvent` now forbids a `.` in `step`
        (`journal.py::JournalEvent._forbid_qualified_task_id`), so that mismatch can no
        longer be constructed in the first place; this pins the defense at its source
        instead of at `take_snapshot`.
        """
        with pytest.raises(ValidationError):
            _completed('pts_target.run_pts_target', 3600.0)

    def test_stalls_are_detected_by_the_ceiling_when_there_is_no_history(self) -> None:
        """No baseline is the common case on early runs, not a fallback to be left untested."""
        tasks = [TaskInstance(task_id='slow', state='running', start_date=NOW - timedelta(hours=9))]
        snap = take_snapshot(_client(tasks), _journal(), 'unified_pipeline', 'r', NOW)
        assert [s.task_id for s in snap.stalls] == ['slow']
        assert snap.stalls[0].basis == 'ceiling'

    def test_stalls_for_two_mapped_shards_of_the_same_step_are_both_reported(self) -> None:
        """Shard 3 of 40 hanging must not be indistinguishable from — or shadowed by — shard 1's stall."""
        tasks = [TaskInstance(task_id='run_gentropy_variant_annotation', map_index=0, state='running',
                               start_date=NOW - timedelta(hours=9)),
                 TaskInstance(task_id='run_gentropy_variant_annotation', map_index=3, state='running',
                               start_date=NOW - timedelta(hours=9))]
        snap = take_snapshot(_client(tasks), _journal(), 'unified_pipeline', 'r', NOW)
        assert {s.task_id for s in snap.stalls} == {
            'run_gentropy_variant_annotation[0]',
            'run_gentropy_variant_annotation[3]',
        }

    def test_a_healthy_run_reports_no_stalls(self) -> None:
        tasks = [TaskInstance(task_id='ok', state='running', start_date=NOW - timedelta(minutes=5))]
        snap = take_snapshot(_client(tasks), _journal(), 'unified_pipeline', 'r', NOW)
        assert snap.stalls == []

    def test_a_run_with_no_task_instances_yet_is_empty_not_broken(self) -> None:
        """A just-triggered run has no task instances yet — a common state, not an edge case."""
        snap = take_snapshot(_client([]), _journal(), 'unified_pipeline', 'r', NOW)
        assert snap.counts == {}
        assert snap.running == []
        assert snap.failed == []
        assert snap.stalls == []

    def test_the_journal_cursor_is_the_event_count(self) -> None:
        snap = take_snapshot(_client([]), _journal([_completed('a', 1.0)]), 'unified_pipeline', 'r', NOW)
        assert snap.journal_events == 1

    def test_lists_succeeded_task_refs(self) -> None:
        tasks = [TaskInstance(task_id='a', state='success', duration=12.0),
                 TaskInstance(task_id='b', state='failed', duration=5.0)]
        snap = take_snapshot(_client(tasks), _journal(), 'unified_pipeline', 'r', NOW)
        assert snap.succeeded == ['a']

    def test_distinguishes_mapped_succeeded_task_instances(self) -> None:
        """The two Google Batch steps expand N task instances under one shared task_id.

        Without map_index, `succeeded` would list the same id twice with nothing to
        tell the shards apart — the same trap `running` and `failed` already guard
        against.
        """
        tasks = [TaskInstance(task_id='run_gentropy_variant_annotation', map_index=0,
                               state='success', duration=10.0),
                 TaskInstance(task_id='run_gentropy_variant_annotation', map_index=1,
                               state='success', duration=20.0)]
        snap = take_snapshot(_client(tasks), _journal(), 'unified_pipeline', 'r', NOW)
        assert snap.succeeded == ['run_gentropy_variant_annotation[0]', 'run_gentropy_variant_annotation[1]']

    def test_records_each_task_instances_duration_keyed_by_ref(self) -> None:
        tasks = [TaskInstance(task_id='a', state='success', duration=12.5),
                 TaskInstance(task_id='b', state='failed', duration=5.0)]
        snap = take_snapshot(_client(tasks), _journal(), 'unified_pipeline', 'r', NOW)
        assert snap.durations == {'a': 12.5, 'b': 5.0}

    def test_a_task_instance_with_no_duration_is_absent_from_durations(self) -> None:
        """A running task has no duration yet — `TaskInstance.duration` is None until it finishes.

        The dict must not carry a fabricated `None` or `0.0` for it: either would make
        an empty baseline (no `step_completed` history yet) indistinguishable from a
        step that genuinely ran for zero seconds.
        """
        tasks = [TaskInstance(task_id='a', state='running', start_date=NOW, duration=None)]
        snap = take_snapshot(_client(tasks), _journal(), 'unified_pipeline', 'r', NOW)
        assert snap.durations == {}

    def test_durations_are_keyed_by_the_mapped_ref_not_the_bare_task_id(self) -> None:
        """Two shards sharing a task_id must not collapse onto one duration in the dict."""
        tasks = [TaskInstance(task_id='t', map_index=0, state='success', duration=10.0),
                 TaskInstance(task_id='t', map_index=1, state='success', duration=99.0)]
        snap = take_snapshot(_client(tasks), _journal(), 'unified_pipeline', 'r', NOW)
        assert snap.durations == {'t[0]': 10.0, 't[1]': 99.0}

    def test_records_each_task_instances_try_number_keyed_by_ref(self) -> None:
        tasks = [TaskInstance(task_id='a', state='failed', try_number=1),
                 TaskInstance(task_id='b', state='failed', try_number=2)]
        snap = take_snapshot(_client(tasks), _journal(), 'unified_pipeline', 'r', NOW)
        assert snap.try_numbers == {'a': 1, 'b': 2}

    def test_try_numbers_are_keyed_by_the_mapped_ref_not_the_bare_task_id(self) -> None:
        """A re-run shard and its sibling must not collapse onto one try_number either."""
        tasks = [TaskInstance(task_id='t', map_index=0, state='failed', try_number=1),
                 TaskInstance(task_id='t', map_index=1, state='failed', try_number=2)]
        snap = take_snapshot(_client(tasks), _journal(), 'unified_pipeline', 'r', NOW)
        assert snap.try_numbers == {'t[0]': 1, 't[1]': 2}

    def test_run_started_comes_from_the_dag_run(self) -> None:
        snap = take_snapshot(_client([]), _journal(), 'unified_pipeline', 'r', NOW)
        assert snap.run_started == NOW

    def test_run_stall_is_populated_when_the_run_is_stuck(self) -> None:
        """A task with no Airflow state at all is counted pending, per `_PENDING` — nothing is active."""
        tasks = [TaskInstance(task_id='a')]
        snap = take_snapshot(_client(tasks), _journal(), 'unified_pipeline', 'r', NOW)
        assert snap.run_stall is not None
        assert snap.run_stall.reason == 'stuck_trigger'
        assert snap.run_stall.pending == 1

    def test_run_stall_is_none_for_a_healthy_run(self) -> None:
        tasks = [TaskInstance(task_id='ok', state='running', start_date=NOW - timedelta(minutes=5))]
        snap = take_snapshot(_client(tasks), _journal(), 'unified_pipeline', 'r', NOW)
        assert snap.run_stall is None

    def test_run_stall_reads_the_same_journal_the_per_task_baseline_does(self) -> None:
        """`take_snapshot` must thread the *same* heartbeat history into `run_stalled` it reads for `stalled`.

        One active, already-flagged task plus a journal already carrying enough
        heartbeats for `'no_progress'` to fire on the very first wakeup that computes it.

        Unlike the `TestRunStalled` cases above, this one deliberately does NOT inject a
        threshold: `take_snapshot` is the production wiring, so the point is that it
        applies the real `_RUN_STALL_WAKEUP_THRESHOLD`. The heartbeat count therefore
        tracks that constant, and `test_supervisor_stall.py` is what keeps the constant
        honest against the deployed cron interval.
        """
        tasks = [TaskInstance(task_id='slow', state='running', start_date=NOW - timedelta(hours=9))]
        journal = _journal(_heartbeats(_RUN_STALL_WAKEUP_THRESHOLD, start=NOW - timedelta(hours=1)))
        snap = take_snapshot(_client(tasks), journal, 'unified_pipeline', 'r', NOW)
        assert snap.run_stall is not None
        assert snap.run_stall.reason == 'no_progress'
        assert snap.run_stall.wakeups == _RUN_STALL_WAKEUP_THRESHOLD


class TestRenderSnapshot:
    def _snapshot(self, **kw: object) -> Snapshot:
        defaults: dict[str, object] = {
            'dag_id': 'unified_pipeline', 'run_id': 'r', 'taken_at': NOW, 'run_state': 'running',
            'counts': {'success': 40, 'running': 2}, 'running': ['a', 'b'], 'failed': [],
            'succeeded': [], 'durations': {}, 'try_numbers': {}, 'stalls': [], 'journal_events': 0,
        }
        defaults.update(kw)
        return Snapshot.model_validate(defaults)

    def test_shows_the_run_and_its_counts(self) -> None:
        out = render_snapshot(self._snapshot())
        assert 'unified_pipeline' in out
        assert '40' in out

    def test_shows_when_the_run_started(self) -> None:
        out = render_snapshot(self._snapshot(run_started=datetime(2026, 7, 21, 14, 0, tzinfo=UTC)))
        assert '2026-07-21 14:00' in out

    def test_omits_the_started_line_when_the_run_has_not_started_yet(self) -> None:
        out = render_snapshot(self._snapshot(run_started=None))
        assert 'started' not in out

    def test_a_stall_is_not_buried_in_the_counts(self) -> None:
        """A stall is a first-class escalation, never a line in a digest.

        Position is pinned, not just presence: a `'STALL' in out` assertion would pass
        even if the line were the hundredth in the report.
        """
        out = render_snapshot(self._snapshot(
            stalls=[StallVerdict(task_id='slow', elapsed=32400.0, threshold=21600.0, basis='ceiling')]
        ))
        assert 'STALL' in out.upper()
        assert 'slow' in out
        lines = out.splitlines()
        stall_index = next(i for i, line in enumerate(lines) if line.startswith('STALL'))
        assert stall_index <= 3, 'a stall must sit right after the header and counts, not require scrolling'

    def test_a_stall_precedes_the_failures(self) -> None:
        """A reader scanning top-down must hit the stall before the failure list."""
        out = render_snapshot(self._snapshot(
            stalls=[StallVerdict(task_id='slow', elapsed=32400.0, threshold=21600.0, basis='ceiling')],
            failed=['run_pts_target'],
        ))
        lines = out.splitlines()
        stall_index = next(i for i, line in enumerate(lines) if line.startswith('STALL'))
        failed_index = next(i for i, line in enumerate(lines) if line.startswith('failed:'))
        assert stall_index < failed_index

    def test_failures_are_listed(self) -> None:
        out = render_snapshot(self._snapshot(failed=['run_pts_target']))
        assert 'run_pts_target' in out

    def test_stall_lines_distinguish_mapped_instances(self) -> None:
        """Two STALL lines for the same task_id, with nothing telling them apart, is the bug."""
        out = render_snapshot(self._snapshot(stalls=[
            StallVerdict(task_id='run_gentropy_variant_annotation[0]', elapsed=32400.0,
                        threshold=21600.0, basis='ceiling'),
            StallVerdict(task_id='run_gentropy_variant_annotation[3]', elapsed=32400.0,
                        threshold=21600.0, basis='ceiling'),
        ]))
        stall_lines = [line for line in out.splitlines() if line.startswith('STALL')]
        assert len(stall_lines) == 2
        assert stall_lines[0] != stall_lines[1]
        assert 'run_gentropy_variant_annotation[0]' in stall_lines[0]
        assert 'run_gentropy_variant_annotation[3]' in stall_lines[1]

    def test_a_run_with_no_tasks_renders_cleanly(self) -> None:
        """A just-triggered run has no task instances yet — a common state, not an edge case."""
        out = render_snapshot(self._snapshot(counts={}, running=[], failed=[], stalls=[]))
        assert 'unified_pipeline' in out
        assert out.splitlines()[-1] != ''

    def test_no_run_stall_renders_no_run_stall_line(self) -> None:
        out = render_snapshot(self._snapshot())
        assert 'RUN STALL' not in out

    def test_a_run_stall_is_shown_and_names_its_reason(self) -> None:
        out = render_snapshot(self._snapshot(run_stall=RunStallVerdict(reason='stuck_trigger', pending=4)))
        assert 'RUN STALL' in out
        assert '4' in out

    def test_a_run_stall_precedes_the_per_task_stalls(self) -> None:
        """The run-level verdict is the bigger escalation and must not require scrolling past per-task ones."""
        out = render_snapshot(self._snapshot(
            run_stall=RunStallVerdict(reason='stuck_trigger', pending=4),
            stalls=[StallVerdict(task_id='slow', elapsed=32400.0, threshold=21600.0, basis='ceiling')],
        ))
        lines = out.splitlines()
        run_stall_index = next(i for i, line in enumerate(lines) if line.startswith('RUN STALL'))
        stall_index = next(i for i, line in enumerate(lines) if line.startswith('STALL'))
        assert run_stall_index < stall_index

    def test_a_no_progress_run_stall_names_the_wakeups_and_active_tasks(self) -> None:
        out = render_snapshot(self._snapshot(
            run_stall=RunStallVerdict(reason='no_progress', wakeups=6, active_tasks=2)
        ))
        assert '6' in out
        assert '2' in out
