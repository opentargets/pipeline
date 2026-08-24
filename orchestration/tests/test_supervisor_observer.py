"""Tests for deciding what a wakeup has not already reported."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from orchestration.supervisor.journal import JournalEvent
from orchestration.supervisor.observer import Observation, observe
from orchestration.supervisor.snapshot import Snapshot
from orchestration.supervisor.stall import StallVerdict

NOW = datetime(2026, 7, 21, 20, 0, tzinfo=UTC)


def _snapshot(**kw: object) -> Snapshot:
    defaults: dict[str, object] = {
        'dag_id': 'unified_pipeline', 'run_id': 'r', 'taken_at': NOW, 'run_state': 'running',
        'counts': {}, 'running': [], 'failed': [], 'succeeded': [], 'durations': {},
        'stalls': [], 'journal_events': 0,
    }
    defaults.update(kw)
    return Snapshot.model_validate(defaults)


def _event(event_type: str, step: str | None = None, map_index: int | None = None) -> JournalEvent:
    return JournalEvent(event_type=event_type, step=step, map_index=map_index, at=NOW)


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


class TestObservationIsEmpty:
    def test_a_fresh_observation_is_empty(self) -> None:
        assert Observation().is_empty is True

    def test_a_failure_makes_it_not_empty(self) -> None:
        assert observe(_snapshot(failed=['a']), []).is_empty is False

    def test_a_stall_makes_it_not_empty(self) -> None:
        assert observe(_snapshot(stalls=[_stall('slow')]), []).is_empty is False

    def test_a_finished_run_alone_makes_it_not_empty(self) -> None:
        """run_finished can carry news on its own, with both lists empty."""
        obs = observe(_snapshot(run_state='success'), [])
        assert obs.failed == []
        assert obs.stalled == []
        assert obs.is_empty is False

    def test_nothing_new_at_all_is_empty(self) -> None:
        assert observe(_snapshot(run_state='running'), []).is_empty is True


class TestObserveComposesAllThree:
    def test_a_failure_a_stall_and_a_finished_run_all_surface_together(self) -> None:
        snap = _snapshot(
            run_state='failed',
            failed=['pts_target.run_pts_target'],
            stalls=[_stall('slow')],
        )
        obs = observe(snap, [])
        assert [f.step for f in obs.failed] == ['pts_target']
        assert [s.step for s in obs.stalled] == ['slow']
        assert obs.run_finished == 'failed'
