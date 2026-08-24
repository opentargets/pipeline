"""Tests for stall detection and snapshot assembly."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from orchestration.supervisor.airflow import DagRun, TaskInstance
from orchestration.supervisor.journal import JournalEvent
from orchestration.supervisor.snapshot import Snapshot, render_snapshot, take_snapshot
from orchestration.supervisor.stall import StallVerdict, baseline_from_journal, stalled

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


class TestRenderSnapshot:
    def _snapshot(self, **kw: object) -> Snapshot:
        defaults: dict[str, object] = {
            'dag_id': 'unified_pipeline', 'run_id': 'r', 'taken_at': NOW, 'run_state': 'running',
            'counts': {'success': 40, 'running': 2}, 'running': ['a', 'b'], 'failed': [],
            'stalls': [], 'journal_events': 0,
        }
        defaults.update(kw)
        return Snapshot.model_validate(defaults)

    def test_shows_the_run_and_its_counts(self) -> None:
        out = render_snapshot(self._snapshot())
        assert 'unified_pipeline' in out
        assert '40' in out

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
