"""Tests for rendering a wakeup as a GitHub issue comment."""

from __future__ import annotations

from datetime import UTC, datetime

from orchestration.supervisor.diff import ColumnChange, DatasetDiff
from orchestration.supervisor.observer import Observation, StepCompletion, StepFailure, StepStall
from orchestration.supervisor.report import render_comment
from orchestration.supervisor.snapshot import Snapshot
from orchestration.supervisor.stall import RunStallVerdict

NOW = datetime(2026, 8, 24, 15, 30, tzinfo=UTC)


def _snapshot(**kw: object) -> Snapshot:
    defaults: dict[str, object] = {
        'dag_id': 'unified_pipeline', 'run_id': 'manual__2026-08-24', 'taken_at': NOW, 'run_state': 'running',
        'counts': {}, 'running': [], 'failed': [], 'succeeded': [], 'durations': {},
        'stalls': [], 'journal_events': 0,
    }
    defaults.update(kw)
    return Snapshot.model_validate(defaults)


def _observation(**kw: object) -> Observation:
    return Observation.model_validate(kw)


class TestNothingNewRendersNoComment:
    def test_an_empty_observation_with_no_diff_renders_none(self) -> None:
        assert render_comment(_observation(), _snapshot()) is None

    def test_the_sentinel_is_none_not_an_empty_string(self) -> None:
        """Assert the sentinel directly.

        A broken renderer that returns '' for "nothing new" would also pass a bare
        falsy check. Pin the actual sentinel so that regression is caught.
        """
        result = render_comment(_observation(), _snapshot())
        assert result is None
        assert result != ''

    def test_a_diff_that_ran_with_zero_datasets_still_renders(self) -> None:
        """A ran-but-empty diff is news too.

        An empty `diffs` list means a comparison ran and found nothing to compare —
        that is itself news, distinct from no comparison having run at all (`None`).
        """
        result = render_comment(_observation(), _snapshot(), diffs=[])
        assert result is not None
        assert '0 dataset(s) compared' in result


class TestFailures:
    def test_a_failed_step_is_named(self) -> None:
        obs = _observation(failed=[StepFailure(ref='pts_target.run_pts_target', step='pts_target', map_index=-1)])
        result = render_comment(obs, _snapshot())
        assert result is not None
        assert '**Failed**' in result
        assert '`pts_target`' in result

    def test_every_failure_is_listed(self) -> None:
        obs = _observation(failed=[
            StepFailure(ref='pts_target.run_pts_target', step='pts_target', map_index=-1),
            StepFailure(ref='pts_disease.run_pts_disease', step='pts_disease', map_index=-1),
        ])
        result = render_comment(obs, _snapshot())
        assert result is not None
        assert '`pts_target`' in result
        assert '`pts_disease`' in result

    def test_a_mapped_shard_is_identified_by_its_index_not_the_bare_step(self) -> None:
        obs = _observation(failed=[
            StepFailure(
                ref='gentropy_variant_annotation.gentropy_variant_annotation_batch_jobs.run_gentropy_variant_annotation[3]',
                step='gentropy_variant_annotation', map_index=3,
            ),
        ])
        result = render_comment(obs, _snapshot())
        assert result is not None
        assert '`gentropy_variant_annotation[3]`' in result
        assert '`gentropy_variant_annotation`' not in result

    def test_an_unmapped_failure_is_not_qualified_with_an_index(self) -> None:
        obs = _observation(failed=[StepFailure(ref='pts_target.run_pts_target', step='pts_target', map_index=-1)])
        result = render_comment(obs, _snapshot())
        assert result is not None
        assert '[-1]' not in result


class TestTaskIdentity:
    """F5: two tasks in the same group must not render identically.

    `_label` alone collapses every task in a group onto the bare step name, so a
    config-upload failure and the run task's own failure — completely different
    failure modes calling for completely different responses — used to be
    indistinguishable in the comment.
    """

    def test_two_tasks_in_the_same_group_render_distinguishably(self) -> None:
        obs = _observation(failed=[
            StepFailure(ref='pts_target.upload_config_pts_target', step='pts_target', map_index=-1),
            StepFailure(ref='pts_target.run_pts_target', step='pts_target', map_index=-1),
        ])
        result = render_comment(obs, _snapshot())
        assert result is not None
        lines = [line for line in result.split('\n') if line.startswith('- ')]
        assert len(lines) == 2
        assert lines[0] != lines[1]

    def test_a_non_run_task_failure_shows_its_full_ref(self) -> None:
        obs = _observation(failed=[
            StepFailure(ref='pts_target.upload_config_pts_target', step='pts_target', map_index=-1),
        ])
        result = render_comment(obs, _snapshot())
        assert result is not None
        assert 'pts_target.upload_config_pts_target' in result

    def test_the_run_tasks_own_failure_stays_terse(self) -> None:
        """The common case must not get noisier from this fix."""
        obs = _observation(failed=[
            StepFailure(ref='pts_target.run_pts_target', step='pts_target', map_index=-1),
        ])
        result = render_comment(obs, _snapshot())
        assert result is not None
        assert result.endswith('- `pts_target`')
        assert 'run_pts_target' not in result

    def test_a_mapped_shards_run_task_failure_is_still_identifiable_by_shard_alone(self) -> None:
        """The run task's own shards stay terse too — disambiguation is orthogonal to sharding."""
        obs = _observation(failed=[
            StepFailure(
                ref='gentropy_variant_annotation.gentropy_variant_annotation_batch_jobs.'
                    'run_gentropy_variant_annotation[3]',
                step='gentropy_variant_annotation', map_index=3,
            ),
        ])
        result = render_comment(obs, _snapshot())
        assert result is not None
        assert result.endswith('- `gentropy_variant_annotation[3]`')

    def test_a_non_run_task_stall_shows_its_full_ref(self) -> None:
        obs = _observation(stalled=[
            StepStall(ref='pts_target.diff_pts_target', step='pts_target', map_index=-1,
                      elapsed=32400.0, threshold=21600.0, basis='ceiling'),
        ])
        result = render_comment(obs, _snapshot())
        assert result is not None
        assert 'pts_target.diff_pts_target' in result

    def test_a_non_run_task_and_the_run_task_stalling_together_are_distinguishable(self) -> None:
        obs = _observation(stalled=[
            StepStall(ref='pts_target.diff_pts_target', step='pts_target', map_index=-1,
                      elapsed=32400.0, threshold=21600.0, basis='ceiling'),
            StepStall(ref='pts_target.run_pts_target', step='pts_target', map_index=-1,
                      elapsed=32400.0, threshold=21600.0, basis='ceiling'),
        ])
        result = render_comment(obs, _snapshot())
        assert result is not None
        lines = [line for line in result.split('\n') if line.startswith('- ')]
        assert len(lines) == 2
        assert lines[0] != lines[1]


class TestStalls:
    def test_a_stall_states_elapsed_against_threshold(self) -> None:
        obs = _observation(stalled=[
            StepStall(ref='pts_target.run_pts_target', step='pts_target', map_index=-1,
                      elapsed=32400.0, threshold=21600.0, basis='ceiling'),
        ])
        result = render_comment(obs, _snapshot())
        assert result is not None
        assert '**Stalled**' in result
        assert '9h' in result
        assert '6h' in result

    def test_a_ceiling_stall_names_the_ceiling_rule(self) -> None:
        obs = _observation(stalled=[
            StepStall(ref='pts_target.run_pts_target', step='pts_target', map_index=-1,
                      elapsed=32400.0, threshold=21600.0, basis='ceiling'),
        ])
        result = render_comment(obs, _snapshot())
        assert result is not None
        assert 'ceiling' in result
        assert 'history' not in result

    def test_a_history_stall_is_rendered_distinctly_from_a_ceiling_stall(self) -> None:
        """History firing is unusual and must read as such.

        `stall.py` says the history rule is nearly unreachable, so it firing at all is
        unusual and must not read as an implementation detail alongside the ceiling.
        """
        ceiling_obs = _observation(stalled=[
            StepStall(ref='pts_target.run_pts_target', step='pts_target', map_index=-1,
                      elapsed=32400.0, threshold=21600.0, basis='ceiling'),
        ])
        history_obs = _observation(stalled=[
            StepStall(ref='pts_target.run_pts_target', step='pts_target', map_index=-1,
                      elapsed=32400.0, threshold=21600.0, basis='history'),
        ])
        ceiling_result = render_comment(ceiling_obs, _snapshot())
        history_result = render_comment(history_obs, _snapshot())
        assert ceiling_result is not None
        assert history_result is not None
        assert ceiling_result != history_result
        assert 'unusual' in history_result
        assert 'unusual' not in ceiling_result

    def test_a_stalled_mapped_shard_is_identified(self) -> None:
        """`ref` must be the real, group-qualified Airflow task_id.

        `TaskInstance.ref` is always group-qualified, and `gentropy_l2g_prediction`
        nests its run task two groups deep (`cluster: false`) — an unqualified ref
        here would not exercise the real shape, and would silently take
        `_identify`'s non-run-task branch instead (F5).
        """
        obs = _observation(stalled=[
            StepStall(
                ref='gentropy_l2g_prediction.gentropy_l2g_prediction_batch_jobs.run_gentropy_l2g_prediction[7]',
                step='gentropy_l2g_prediction', map_index=7,
                elapsed=32400.0, threshold=21600.0, basis='ceiling',
            ),
        ])
        result = render_comment(obs, _snapshot())
        assert result is not None
        assert result.count('- ') == 1
        assert '- `gentropy_l2g_prediction[7]` — running' in result
        assert 'task `gentropy_l2g_prediction' not in result

    def test_every_stall_is_listed(self) -> None:
        obs = _observation(stalled=[
            StepStall(ref='pts_target.run_pts_target', step='pts_target', map_index=-1,
                      elapsed=32400.0, threshold=21600.0, basis='ceiling'),
            StepStall(ref='pts_disease.run_pts_disease', step='pts_disease', map_index=-1,
                      elapsed=25000.0, threshold=21600.0, basis='ceiling'),
        ])
        result = render_comment(obs, _snapshot())
        assert result is not None
        assert '`pts_target`' in result
        assert '`pts_disease`' in result


class TestCompletions:
    def test_a_completion_states_a_human_readable_duration_not_raw_seconds(self) -> None:
        obs = _observation(completed=[
            StepCompletion(ref='pts_target.run_pts_target', step='pts_target', map_index=-1, duration=252.0),
        ])
        result = render_comment(obs, _snapshot())
        assert result is not None
        assert '4m12s' in result
        assert '252' not in result

    def test_a_large_duration_reads_in_hours(self) -> None:
        obs = _observation(completed=[
            StepCompletion(ref='pts_target.run_pts_target', step='pts_target', map_index=-1, duration=18000.0),
        ])
        result = render_comment(obs, _snapshot())
        assert result is not None
        assert '5h' in result

    def test_a_completed_mapped_shard_is_identified(self) -> None:
        """`ref` is fully group-qualified — see `TestStalls`' equivalent test."""
        obs = _observation(completed=[
            StepCompletion(
                ref='gentropy_variant_annotation.gentropy_variant_annotation_batch_jobs.'
                    'run_gentropy_variant_annotation[12]',
                step='gentropy_variant_annotation', map_index=12, duration=600.0,
            ),
        ])
        result = render_comment(obs, _snapshot())
        assert result is not None
        assert result.endswith('- `gentropy_variant_annotation[12]` finished in 10m00s')

    def test_every_completion_is_listed(self) -> None:
        obs = _observation(completed=[
            StepCompletion(ref='pts_target.run_pts_target', step='pts_target', map_index=-1, duration=120.0),
            StepCompletion(ref='pts_disease.run_pts_disease', step='pts_disease', map_index=-1, duration=180.0),
            StepCompletion(
                ref='gentropy_variant_annotation.gentropy_variant_annotation_batch_jobs.'
                    'run_gentropy_variant_annotation[5]',
                step='gentropy_variant_annotation', map_index=5, duration=300.0,
            ),
        ])
        result = render_comment(obs, _snapshot())
        assert result is not None
        assert '`pts_target`' in result
        assert '`pts_disease`' in result
        assert '`gentropy_variant_annotation[5]`' in result
        assert 'task `gentropy_variant_annotation' not in result


class TestRunFinished:
    def test_a_run_level_completion_reads_differently_from_a_step_completion(self) -> None:
        obs = _observation(
            completed=[StepCompletion(ref='pts_target.run_pts_target', step='pts_target', map_index=-1,
                                       duration=120.0)],
            run_finished='success',
        )
        result = render_comment(obs, _snapshot())
        assert result is not None
        run_section = [line for line in result.split('\n\n') if 'Run succeeded' in line]
        step_section = [line for line in result.split('\n\n') if '**Completed**' in line]
        assert run_section, 'expected a distinct run-level section'
        assert step_section, 'expected a distinct step-completion section'
        assert run_section[0] != step_section[0]

    def test_a_successful_run_says_so(self) -> None:
        obs = _observation(run_finished='success')
        result = render_comment(obs, _snapshot())
        assert result is not None
        assert 'succeeded' in result
        assert 'FAILED' not in result

    def test_a_failed_run_says_so_distinctly_from_success(self) -> None:
        obs = _observation(run_finished='failed')
        result = render_comment(obs, _snapshot())
        assert result is not None
        assert 'FAILED' in result

    def test_run_finished_names_the_run(self) -> None:
        obs = _observation(run_finished='success')
        result = render_comment(obs, _snapshot(run_id='manual__2026-08-24'))
        assert result is not None
        assert 'manual__2026-08-24' in result

    def test_run_finished_is_not_rendered_when_none(self) -> None:
        obs = _observation(failed=[StepFailure(ref='pts_target.run_pts_target', step='pts_target', map_index=-1)])
        result = render_comment(obs, _snapshot())
        assert result is not None
        assert 'Run succeeded' not in result
        assert 'Run FAILED' not in result


class TestRunStall:
    def test_a_run_stall_gets_its_own_heading(self) -> None:
        obs = _observation(run_stall=RunStallVerdict(reason='stuck_trigger', pending=4))
        result = render_comment(obs, _snapshot())
        assert result is not None
        assert 'Run stalled' in result

    def test_a_stuck_trigger_names_the_pending_count(self) -> None:
        obs = _observation(run_stall=RunStallVerdict(reason='stuck_trigger', pending=4))
        result = render_comment(obs, _snapshot())
        assert result is not None
        assert '4' in result
        assert 'pending' in result

    def test_a_no_progress_verdict_names_the_wakeups_and_active_tasks(self) -> None:
        obs = _observation(run_stall=RunStallVerdict(reason='no_progress', wakeups=6, active_tasks=2))
        result = render_comment(obs, _snapshot())
        assert result is not None
        assert '6' in result
        assert '2' in result

    def test_no_run_stall_renders_no_run_stall_section(self) -> None:
        obs = _observation(failed=[StepFailure(ref='pts_target.run_pts_target', step='pts_target', map_index=-1)])
        result = render_comment(obs, _snapshot())
        assert result is not None
        assert 'Run stalled' not in result

    def test_a_run_stall_precedes_failures_and_stalls(self) -> None:
        obs = _observation(
            run_stall=RunStallVerdict(reason='stuck_trigger', pending=4),
            failed=[StepFailure(ref='pts_target.run_pts_target', step='pts_target', map_index=-1)],
        )
        result = render_comment(obs, _snapshot())
        assert result is not None
        assert result.index('Run stalled') < result.index('**Failed**')

    def test_a_run_stall_alone_is_enough_to_render_a_comment(self) -> None:
        """`Observation.is_empty` must not swallow a run stall with nothing else new."""
        obs = _observation(run_stall=RunStallVerdict(reason='stuck_trigger', pending=1))
        assert render_comment(obs, _snapshot()) is not None


class TestDiffs:
    def test_no_material_differences_says_so_plainly(self) -> None:
        diffs = [DatasetDiff(dataset='output/disease', side='both', run_rows=100, reference_rows=100,
                              run_bytes=1000, reference_bytes=1000, run_files=1, reference_files=1)]
        result = render_comment(_observation(), _snapshot(), diffs=diffs)
        assert result is not None
        assert 'No material differences' in result

    def test_a_material_row_change_is_reported(self) -> None:
        diffs = [DatasetDiff(dataset='output/disease', side='both', run_rows=100, reference_rows=10,
                              run_bytes=1000, reference_bytes=1000, run_files=1, reference_files=1)]
        result = render_comment(_observation(), _snapshot(), diffs=diffs)
        assert result is not None
        assert '`output/disease`' in result
        assert '10 -> 100' in result

    def test_a_one_sided_dataset_is_named(self) -> None:
        diffs = [DatasetDiff(dataset='output/new_thing', side='run_only', run_bytes=500, run_files=1)]
        result = render_comment(_observation(), _snapshot(), diffs=diffs)
        assert result is not None
        assert '`output/new_thing`' in result
        assert 'the run only' in result

    def test_a_schema_change_is_always_reported_regardless_of_threshold(self) -> None:
        diffs = [DatasetDiff(
            dataset='output/disease', side='both', run_rows=100, reference_rows=100,
            run_bytes=1000, reference_bytes=1000, run_files=1, reference_files=1,
            columns=[ColumnChange(column='newCol', kind='added', run_type='string')],
        )]
        result = render_comment(_observation(), _snapshot(), diffs=diffs)
        assert result is not None
        assert 'newCol' in result

    def test_no_diffs_argument_renders_no_dataset_section(self) -> None:
        obs = _observation(failed=[StepFailure(ref='pts_target.run_pts_target', step='pts_target', map_index=-1)])
        result = render_comment(obs, _snapshot())
        assert result is not None
        assert 'Dataset comparison' not in result


class TestRepeatAttempts:
    def test_a_repeat_failure_says_so(self) -> None:
        obs = _observation(failed=[
            StepFailure(ref='pts_target.run_pts_target', step='pts_target', map_index=-1, try_number=3),
        ])
        result = render_comment(obs, _snapshot())
        assert result is not None
        assert 'attempt 3' in result

    def test_a_first_attempt_failure_does_not_mention_an_attempt(self) -> None:
        """The universal ordinary case must not render as if it were news.

        try_number=1 is the universal ordinary case (max_tries is 0 throughout this
        pipeline) and must not render as if it were news.
        """
        obs = _observation(failed=[
            StepFailure(ref='pts_target.run_pts_target', step='pts_target', map_index=-1, try_number=1),
        ])
        result = render_comment(obs, _snapshot())
        assert result is not None
        assert 'attempt' not in result

    def test_an_unknown_try_number_does_not_mention_an_attempt(self) -> None:
        """None must render exactly like 1, and never as a fabricated attempt.

        try_number=None must render exactly like try_number=1 — nothing — not as
        'attempt None' and not by defaulting silently to 'attempt 1'.
        """
        obs = _observation(failed=[
            StepFailure(ref='pts_target.run_pts_target', step='pts_target', map_index=-1, try_number=None),
        ])
        result = render_comment(obs, _snapshot())
        assert result is not None
        assert 'attempt' not in result
        assert 'None' not in result

    def test_a_repeat_stall_says_so(self) -> None:
        obs = _observation(stalled=[
            StepStall(ref='pts_target.run_pts_target', step='pts_target', map_index=-1, try_number=2,
                      elapsed=32400.0, threshold=21600.0, basis='ceiling'),
        ])
        result = render_comment(obs, _snapshot())
        assert result is not None
        assert 'attempt 2' in result

    def test_a_repeat_completion_says_so(self) -> None:
        obs = _observation(completed=[
            StepCompletion(ref='pts_target.run_pts_target', step='pts_target', map_index=-1, try_number=4,
                            duration=252.0),
        ])
        result = render_comment(obs, _snapshot())
        assert result is not None
        assert 'attempt 4' in result

    def test_first_attempt_stalls_and_completions_do_not_mention_an_attempt(self) -> None:
        obs = _observation(
            stalled=[StepStall(ref='pts_target.run_pts_target', step='pts_target', map_index=-1, try_number=1,
                                elapsed=32400.0, threshold=21600.0, basis='ceiling')],
            completed=[StepCompletion(ref='pts_disease.run_pts_disease', step='pts_disease', map_index=-1,
                                       try_number=None, duration=120.0)],
        )
        result = render_comment(obs, _snapshot())
        assert result is not None
        assert 'attempt' not in result


class TestSectionOrdering:
    def test_run_finished_leads_before_failures(self) -> None:
        obs = _observation(
            failed=[StepFailure(ref='pts_target.run_pts_target', step='pts_target', map_index=-1)],
            run_finished='failed',
        )
        result = render_comment(obs, _snapshot())
        assert result is not None
        assert result.index('Run FAILED') < result.index('**Failed**')
