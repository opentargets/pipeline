"""Tests for deciding what a wakeup has not already reported."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from orchestration.supervisor.journal import JournalEvent, heartbeat_event
from orchestration.supervisor.observer import Observation, observe
from orchestration.supervisor.snapshot import Snapshot
from orchestration.supervisor.stall import RunStallVerdict, StallVerdict

NOW = datetime(2026, 7, 21, 20, 0, tzinfo=UTC)


def _snapshot(**kw: object) -> Snapshot:
    defaults: dict[str, object] = {
        'dag_id': 'unified_pipeline', 'run_id': 'r', 'taken_at': NOW, 'run_state': 'running',
        'counts': {}, 'running': [], 'failed': [], 'succeeded': [], 'durations': {}, 'try_numbers': {},
        'stalls': [], 'journal_events': 0,
    }
    defaults.update(kw)
    return Snapshot.model_validate(defaults)


def _event(
    event_type: str, step: str | None = None, map_index: int | None = None, try_number: int | None = None,
) -> JournalEvent:
    return JournalEvent(event_type=event_type, step=step, map_index=map_index, try_number=try_number, at=NOW)


def _stall(task_id: str, basis: Literal['history', 'ceiling'] = 'ceiling') -> StallVerdict:
    return StallVerdict(task_id=task_id, elapsed=32400.0, threshold=21600.0, basis=basis)


class TestObserveFailures:
    def test_a_first_wakeup_reports_every_failure(self) -> None:
        snap = _snapshot(failed=['pts_target.run_pts_target', 'pts_disease.run_pts_disease'])
        obs = observe(snap, [])
        assert {f.step for f in obs.failed} == {'pts_target', 'pts_disease'}

    def test_a_second_wakeup_reports_none_of_the_same_failures_again(self) -> None:
        snap = _snapshot(failed=['pts_target.run_pts_target'])
        already = [_event('step_failed', step='pts_target')]
        assert observe(snap, already).failed == []

    def test_a_newly_failed_step_is_reported_alongside_an_already_known_one(self) -> None:
        snap = _snapshot(failed=['pts_target.run_pts_target', 'pts_disease.run_pts_disease'])
        already = [_event('step_failed', step='pts_target')]
        obs = observe(snap, already)
        assert [f.step for f in obs.failed] == ['pts_disease']

    def test_a_failure_carries_its_ref_step_and_map_index(self) -> None:
        obs = observe(_snapshot(failed=['pts_target.run_pts_target']), [])
        assert obs.failed[0].ref == 'pts_target.run_pts_target'
        assert obs.failed[0].step == 'pts_target'
        assert obs.failed[0].map_index == -1

    def test_two_failed_shards_of_one_mapped_step_both_surface(self) -> None:
        snap = _snapshot(failed=['run_gentropy_variant_annotation[0]', 'run_gentropy_variant_annotation[3]'])
        obs = observe(snap, [])
        assert {f.ref for f in obs.failed} == {
            'run_gentropy_variant_annotation[0]',
            'run_gentropy_variant_annotation[3]',
        }

    def test_a_failed_shard_already_known_does_not_swallow_its_sibling(self) -> None:
        """The map_index trap: shard 3 failing must not be mistaken for shard 0's already-known failure."""
        snap = _snapshot(failed=['run_gentropy_variant_annotation[0]', 'run_gentropy_variant_annotation[3]'])
        already = [_event('step_failed', step='run_gentropy_variant_annotation', map_index=0)]
        obs = observe(snap, already)
        assert [f.ref for f in obs.failed] == ['run_gentropy_variant_annotation[3]']

    def test_no_failures_reports_none(self) -> None:
        assert observe(_snapshot(failed=[]), []).failed == []

    def test_a_failure_carries_its_try_number(self) -> None:
        snap = _snapshot(failed=['pts_target.run_pts_target'], try_numbers={'pts_target.run_pts_target': 1})
        assert observe(snap, []).failed[0].try_number == 1

    def test_a_repeated_failure_after_a_rerun_is_reported_again(self) -> None:
        """The defect: a step that fails, is cleared, runs again and fails again went unreported.

        `try_number` is what makes the second failure a distinct event from the first
        — without it, both attempts produce the same key and the second is discarded
        as a duplicate.
        """
        already = [_event('step_failed', step='pts_target', try_number=1)]
        snap = _snapshot(failed=['pts_target.run_pts_target'], try_numbers={'pts_target.run_pts_target': 2})
        obs = observe(snap, already)
        assert [f.step for f in obs.failed] == ['pts_target']
        assert obs.failed[0].try_number == 2

    def test_a_failure_with_the_same_try_number_already_known_is_not_reported_again(self) -> None:
        already = [_event('step_failed', step='pts_target', try_number=1)]
        snap = _snapshot(failed=['pts_target.run_pts_target'], try_numbers={'pts_target.run_pts_target': 1})
        assert observe(snap, already).failed == []

    def test_the_teams_exact_reproduction_a_rerun_failure_is_not_swallowed_by_the_first(self) -> None:
        """The literal repro from the bug report: `step_failed-pts_target` journalled with no try_number.

        Before this fix, `_already_known` never carried `try_number` on the candidate
        it builds, for *either* attempt — so attempt 1 was journalled as
        `step_failed-pts_target` (no try_number segment at all), and attempt 2's
        candidate, still missing `try_number`, computed that exact same key and was
        discarded as a duplicate. This pins the reported bug in its own words,
        independent of the `try_number`-aware helper tests above.
        """
        journalled_before_the_fix = [_event('step_failed', step='pts_target')]

        wakeup_two = observe(
            _snapshot(failed=['pts_target.run_pts_target'], try_numbers={'pts_target.run_pts_target': 2}),
            journalled_before_the_fix,
        )
        assert [f.step for f in wakeup_two.failed] == ['pts_target']
        assert wakeup_two.failed[0].try_number == 2


class TestObserveStalls:
    def test_a_first_wakeup_reports_every_stall(self) -> None:
        obs = observe(_snapshot(stalls=[_stall('slow')]), [])
        assert [s.step for s in obs.stalled] == ['slow']

    def test_a_second_wakeup_reports_none_of_the_same_stall_again(self) -> None:
        snap = _snapshot(stalls=[_stall('slow')])
        already = [_event('stall_detected', step='slow')]
        assert observe(snap, already).stalled == []

    def test_two_stalled_shards_of_one_mapped_step_both_surface(self) -> None:
        snap = _snapshot(stalls=[
            _stall('run_gentropy_variant_annotation[0]'),
            _stall('run_gentropy_variant_annotation[3]'),
        ])
        obs = observe(snap, [])
        assert {s.ref for s in obs.stalled} == {
            'run_gentropy_variant_annotation[0]',
            'run_gentropy_variant_annotation[3]',
        }

    def test_shard_3_stalling_after_shard_0_is_already_known_is_still_reported(self) -> None:
        """The brief's trap, stated directly: a shard's stall must not be swallowed as a duplicate."""
        snap = _snapshot(stalls=[
            _stall('run_gentropy_variant_annotation[0]'),
            _stall('run_gentropy_variant_annotation[3]'),
        ])
        already = [_event('stall_detected', step='run_gentropy_variant_annotation', map_index=0)]
        obs = observe(snap, already)
        assert [s.ref for s in obs.stalled] == ['run_gentropy_variant_annotation[3]']

    def test_a_stall_carries_elapsed_threshold_and_basis(self) -> None:
        obs = observe(_snapshot(stalls=[_stall('slow', basis='history')]), [])
        assert obs.stalled[0].elapsed == 32400.0
        assert obs.stalled[0].threshold == 21600.0
        assert obs.stalled[0].basis == 'history'

    def test_no_stalls_reports_none(self) -> None:
        assert observe(_snapshot(stalls=[]), []).stalled == []

    def test_a_stall_carries_its_try_number(self) -> None:
        snap = _snapshot(stalls=[_stall('slow')], try_numbers={'slow': 1})
        assert observe(snap, []).stalled[0].try_number == 1

    def test_a_repeated_stall_after_a_rerun_is_reported_again(self) -> None:
        """The same defect as failures: a re-run step stalling again must be a distinct event."""
        already = [_event('stall_detected', step='slow', try_number=1)]
        snap = _snapshot(stalls=[_stall('slow')], try_numbers={'slow': 2})
        obs = observe(snap, already)
        assert [s.step for s in obs.stalled] == ['slow']
        assert obs.stalled[0].try_number == 2

    def test_a_stall_with_the_same_try_number_already_known_is_not_reported_again(self) -> None:
        already = [_event('stall_detected', step='slow', try_number=1)]
        snap = _snapshot(stalls=[_stall('slow')], try_numbers={'slow': 1})
        assert observe(snap, already).stalled == []


class TestObserveCompletions:
    def test_a_first_wakeup_reports_every_completed_run_task(self) -> None:
        snap = _snapshot(succeeded=['pts_target.run_pts_target'], durations={'pts_target.run_pts_target': 3600.0})
        obs = observe(snap, [])
        assert [c.step for c in obs.completed] == ['pts_target']

    def test_a_second_wakeup_reports_none_of_the_same_completion_again(self) -> None:
        snap = _snapshot(succeeded=['pts_target.run_pts_target'], durations={'pts_target.run_pts_target': 3600.0})
        already = [_event('step_completed', step='pts_target')]
        assert observe(snap, already).completed == []

    def test_a_newly_completed_step_is_reported_alongside_an_already_known_one(self) -> None:
        snap = _snapshot(
            succeeded=['pts_target.run_pts_target', 'pts_disease.run_pts_disease'],
            durations={'pts_target.run_pts_target': 3600.0, 'pts_disease.run_pts_disease': 1800.0},
        )
        already = [_event('step_completed', step='pts_target')]
        obs = observe(snap, already)
        assert [c.step for c in obs.completed] == ['pts_disease']

    def test_a_completion_carries_its_ref_step_map_index_and_duration(self) -> None:
        snap = _snapshot(succeeded=['pts_target.run_pts_target'], durations={'pts_target.run_pts_target': 3600.0})
        obs = observe(snap, [])
        assert obs.completed[0].ref == 'pts_target.run_pts_target'
        assert obs.completed[0].step == 'pts_target'
        assert obs.completed[0].map_index == -1
        assert obs.completed[0].duration == 3600.0

    def test_two_completed_shards_of_one_mapped_step_both_surface(self) -> None:
        snap = _snapshot(
            succeeded=['pts_target.run_pts_target[0]', 'pts_target.run_pts_target[3]'],
            durations={'pts_target.run_pts_target[0]': 10.0, 'pts_target.run_pts_target[3]': 20.0},
        )
        obs = observe(snap, [])
        assert {c.ref for c in obs.completed} == {
            'pts_target.run_pts_target[0]',
            'pts_target.run_pts_target[3]',
        }

    def test_a_completed_shard_already_known_does_not_swallow_its_sibling(self) -> None:
        """The map_index trap, for completions this time.

        Shard 3 finishing must not be mistaken for shard 0's already-journalled
        completion.
        """
        snap = _snapshot(
            succeeded=['pts_target.run_pts_target[0]', 'pts_target.run_pts_target[3]'],
            durations={'pts_target.run_pts_target[0]': 10.0, 'pts_target.run_pts_target[3]': 20.0},
        )
        already = [_event('step_completed', step='pts_target', map_index=0)]
        obs = observe(snap, already)
        assert [c.ref for c in obs.completed] == ['pts_target.run_pts_target[3]']

    def test_a_non_run_task_siblings_success_is_not_reported(self) -> None:
        """A `delete_vm_` sibling succeeding is not the step finishing — only the run task counts.

        `stalled` never consults the baseline for anything but a step's own run task
        (`is_run_task`); journalling a sibling's duration into that baseline would be
        dead weight at best, misleading at worst.
        """
        snap = _snapshot(
            succeeded=['pts_target.delete_vm_pts_target'],
            durations={'pts_target.delete_vm_pts_target': 5.0},
        )
        assert observe(snap, []).completed == []

    def test_a_succeeded_run_task_with_no_recorded_duration_is_skipped(self) -> None:
        """A completion with no duration is skipped rather than journalled with a fabricated one.

        Should not happen in practice — Airflow populates `duration` whenever a task
        instance finishes — but a duration-less completion would be useless to the
        baseline anyway.
        """
        snap = _snapshot(succeeded=['pts_target.run_pts_target'], durations={})
        assert observe(snap, []).completed == []

    def test_no_successes_reports_none(self) -> None:
        assert observe(_snapshot(succeeded=[]), []).completed == []

    def test_a_completion_carries_its_try_number(self) -> None:
        snap = _snapshot(
            succeeded=['pts_target.run_pts_target'],
            durations={'pts_target.run_pts_target': 3600.0},
            try_numbers={'pts_target.run_pts_target': 1},
        )
        assert observe(snap, []).completed[0].try_number == 1

    def test_a_completion_after_a_rerun_that_previously_failed_is_reported(self) -> None:
        """The same defect, for a step that failed once and then succeeded on a re-run."""
        already = [_event('step_failed', step='pts_target', try_number=1)]
        snap = _snapshot(
            succeeded=['pts_target.run_pts_target'],
            durations={'pts_target.run_pts_target': 3600.0},
            try_numbers={'pts_target.run_pts_target': 2},
        )
        obs = observe(snap, already)
        assert [c.step for c in obs.completed] == ['pts_target']
        assert obs.completed[0].try_number == 2

    def test_a_completion_with_the_same_try_number_already_known_is_not_reported_again(self) -> None:
        already = [_event('step_completed', step='pts_target', try_number=1)]
        snap = _snapshot(
            succeeded=['pts_target.run_pts_target'],
            durations={'pts_target.run_pts_target': 3600.0},
            try_numbers={'pts_target.run_pts_target': 1},
        )
        assert observe(snap, already).completed == []


class TestObserveRunFinished:
    def test_a_running_run_reports_no_run_finished(self) -> None:
        assert observe(_snapshot(run_state='running'), []).run_finished is None

    def test_a_run_with_no_state_yet_reports_no_run_finished(self) -> None:
        assert observe(_snapshot(run_state=None), []).run_finished is None

    def test_a_successful_run_is_reported(self) -> None:
        assert observe(_snapshot(run_state='success'), []).run_finished == 'success'

    def test_a_failed_run_is_reported_as_such(self) -> None:
        assert observe(_snapshot(run_state='failed'), []).run_finished == 'failed'

    def test_a_finished_run_already_journalled_is_not_reported_again(self) -> None:
        """The requirement stated directly: a finished run is detected once, not on every subsequent wakeup."""
        already = [_event('run_finished')]
        assert observe(_snapshot(run_state='success'), already).run_finished is None

    def test_a_finished_run_stays_unreported_across_several_more_wakeups(self) -> None:
        already = [_event('run_finished')]
        for _ in range(3):
            assert observe(_snapshot(run_state='success'), already).run_finished is None


class TestObserveRunStall:
    def test_no_run_stall_on_the_snapshot_reports_none(self) -> None:
        assert observe(_snapshot(run_stall=None), []).run_stall is None

    def test_a_first_wakeup_reports_a_new_run_stall(self) -> None:
        verdict = RunStallVerdict(reason='stuck_trigger', pending=3)
        obs = observe(_snapshot(run_stall=verdict), [])
        assert obs.run_stall == verdict

    def test_a_run_stall_already_journalled_for_the_same_reason_is_not_reported_again(self) -> None:
        already = [_event('run_stall_detected_stuck_trigger')]
        verdict = RunStallVerdict(reason='stuck_trigger', pending=3)
        assert observe(_snapshot(run_stall=verdict), already).run_stall is None

    def test_a_run_stall_stays_unreported_across_several_more_wakeups(self) -> None:
        """Mirrors `run_finished`'s once-and-never-again idempotency."""
        already = [_event('run_stall_detected_stuck_trigger')]
        verdict = RunStallVerdict(reason='stuck_trigger', pending=3)
        for _ in range(3):
            assert observe(_snapshot(run_stall=verdict), already).run_stall is None

    def test_a_different_reason_is_not_silenced_by_the_first_ones_journal_entry(self) -> None:
        """`'no_progress'` firing later in the same run must not be swallowed by an earlier `'stuck_trigger'`."""
        already = [_event('run_stall_detected_stuck_trigger')]
        verdict = RunStallVerdict(reason='no_progress', wakeups=6, active_tasks=1)
        obs = observe(_snapshot(run_stall=verdict), already)
        assert obs.run_stall == verdict

    def test_heartbeats_in_the_journal_never_make_a_run_stall_verdict_already_known(self) -> None:
        """A journal full of heartbeats (and nothing else) must not be mistaken for a prior report."""
        already = [heartbeat_event(NOW), heartbeat_event(NOW)]
        verdict = RunStallVerdict(reason='stuck_trigger', pending=3)
        obs = observe(_snapshot(run_stall=verdict), already)
        assert obs.run_stall == verdict


class TestObservationIsEmpty:
    def test_a_fresh_observation_is_empty(self) -> None:
        assert Observation().is_empty is True

    def test_a_failure_makes_it_not_empty(self) -> None:
        assert observe(_snapshot(failed=['a']), []).is_empty is False

    def test_a_stall_makes_it_not_empty(self) -> None:
        assert observe(_snapshot(stalls=[_stall('slow')]), []).is_empty is False

    def test_a_completion_makes_it_not_empty(self) -> None:
        snap = _snapshot(succeeded=['pts_target.run_pts_target'], durations={'pts_target.run_pts_target': 3600.0})
        assert observe(snap, []).is_empty is False

    def test_a_finished_run_alone_makes_it_not_empty(self) -> None:
        """run_finished can carry news on its own, with everything else empty."""
        obs = observe(_snapshot(run_state='success'), [])
        assert obs.failed == []
        assert obs.stalled == []
        assert obs.completed == []
        assert obs.is_empty is False

    def test_nothing_new_at_all_is_empty(self) -> None:
        assert observe(_snapshot(run_state='running'), []).is_empty is True

    def test_a_run_stall_alone_makes_it_not_empty(self) -> None:
        """`run_stall` can carry news on its own, exactly like `run_finished` above.

        Without this, `_observation_events` (`cli.py`) would still journal the verdict
        via `run_stall`, but `render_comment` would see an "empty" `Observation` and
        post nothing — the verdict would be recorded and never surfaced to a human.
        """
        verdict = RunStallVerdict(reason='stuck_trigger', pending=3)
        obs = observe(_snapshot(run_state='running', run_stall=verdict), [])
        assert obs.failed == []
        assert obs.stalled == []
        assert obs.completed == []
        assert obs.run_finished is None
        assert obs.is_empty is False

    def test_a_journal_full_of_heartbeats_and_nothing_else_stays_empty(self) -> None:
        """The requirement stated directly: heartbeats must never make a wakeup report itself.

        A wakeup that finds nothing new, against a journal that holds only heartbeats
        from earlier wakeups (and a run that has no run-level stall to report), must
        still be empty — the whole point of a heartbeat is to be silent unless it is
        also evidence for a real verdict.
        """
        already = [heartbeat_event(NOW), heartbeat_event(NOW)]
        assert observe(_snapshot(run_state='running'), already).is_empty is True


class TestObserveComposesAllFour:
    def test_a_failure_a_stall_a_completion_and_a_finished_run_all_surface_together(self) -> None:
        snap = _snapshot(
            run_state='success',
            failed=['pts_target.run_pts_target'],
            stalls=[_stall('slow')],
            succeeded=['pts_disease.run_pts_disease'],
            durations={'pts_disease.run_pts_disease': 300.0},
        )
        obs = observe(snap, [])
        assert [f.step for f in obs.failed] == ['pts_target']
        assert [s.step for s in obs.stalled] == ['slow']
        assert [c.step for c in obs.completed] == ['pts_disease']
        assert obs.run_finished == 'success'
