"""Tests for stall detection and snapshot assembly."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from orchestration.supervisor.airflow import TaskInstance
from orchestration.supervisor.journal import JournalEvent
from orchestration.supervisor.stall import baseline_from_journal, stalled

NOW = datetime(2026, 7, 21, 20, 0, tzinfo=UTC)


def _running(task_id: str = 'run_pts_target', minutes: int = 30) -> TaskInstance:
    return TaskInstance(task_id=task_id, state='running', start_date=NOW - timedelta(minutes=minutes))


def _completed(step: str, duration: float) -> JournalEvent:
    return JournalEvent(event_type='step_completed', step=step, try_number=1, at=NOW,
                        payload={'duration': duration})


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
        assert stalled(_running(minutes=30), {'run_pts_target': 3600.0}, NOW) is None

    def test_a_task_far_past_its_baseline_is_stalled(self) -> None:
        verdict = stalled(_running(minutes=300), {'run_pts_target': 3600.0}, NOW)
        assert verdict is not None
        assert verdict.basis == 'history'

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
