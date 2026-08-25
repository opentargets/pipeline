"""Tests for reading job execution times from a Dataproc-shaped client."""

from __future__ import annotations

import enum
from datetime import UTC, datetime, timedelta
from typing import Any

from orchestration.supervisor.dataproc import (
    JobExecution,
    job_execution,
    job_executions,
    step_for_job_id,
)

_T0 = datetime(2026, 7, 21, 17, 4, 52, tzinfo=UTC)
_PENDING = _T0
_SETUP_DONE = _T0 + timedelta(seconds=1)
_RUNNING = _T0 + timedelta(seconds=2)
_DONE = _T0 + timedelta(minutes=19)


class _RealShapedState(enum.Enum):
    """Stands in for `JobStatus.State`: a real enum whose `str()` is not its name.

    `_state_name` must read `.name` rather than fall back to `str(state)` -- a plain
    string test double would pass either way, since a string's `str()` is itself.
    """

    RUNNING = 2
    DONE = 5
    CANCELLED = 4
    ERROR = 6
    PENDING = 1
    SETUP_DONE = 8


class FakeStatus:
    def __init__(self, state: Any, state_start_time: datetime | None) -> None:
        self.state = state
        self.state_start_time = state_start_time


class FakeReference:
    def __init__(self, job_id: str) -> None:
        self.job_id = job_id


class FakeJob:
    def __init__(
        self,
        job_id: str,
        status: FakeStatus,
        status_history: list[FakeStatus],
        labels: dict[str, str] | None = None,
    ) -> None:
        self.reference = FakeReference(job_id)
        self.status = status
        self.status_history = status_history
        self.labels = labels if labels is not None else {}


def _job(
    job_id: str,
    state: Any,
    ended: datetime | None,
    history: list[FakeStatus],
    labels: dict[str, str] | None = None,
) -> FakeJob:
    return FakeJob(job_id, FakeStatus(state, ended), history, labels)


def _finished_job(job_id: str, state: Any = 'DONE', labels: dict[str, str] | None = None) -> FakeJob:
    """A job that ran to completion: PENDING -> SETUP_DONE -> RUNNING -> `state`."""
    history = [
        FakeStatus('PENDING', _PENDING),
        FakeStatus('SETUP_DONE', _SETUP_DONE),
        FakeStatus('RUNNING', _RUNNING),
    ]
    return _job(job_id, state, _DONE, history, labels)


class FakeClient:
    """Stands in for `JobControllerClient`, recording how it was called."""

    def __init__(self, jobs: list[FakeJob]) -> None:
        self.jobs = jobs
        self.calls: list[dict[str, Any]] = []

    def list_jobs(self, *, project_id: str, region: str, filter: str) -> list[FakeJob]:
        self.calls.append({'project_id': project_id, 'region': region, 'filter': filter})
        return self.jobs


class TestExecutionTime:
    def test_execution_time_is_the_terminal_status_minus_the_running_entry(self) -> None:
        job = _finished_job('up-pts-f5014-pts_drug_molecule-c516h')
        execution = job_execution(job, known_steps=['pts_drug_molecule'])
        assert execution.started == _RUNNING
        assert execution.ended == _DONE
        assert execution.execution_seconds == (_DONE - _RUNNING).total_seconds()

    def test_a_job_with_no_running_entry_yields_none_not_zero(self) -> None:
        """A job that failed during setup never ran, so it has no execution time at all.

        Breaking the `started is not None` guard into treating a missing start as
        `0` would report this job as having executed instantly, which is the
        opposite of the truth: it never executed.
        """
        history = [FakeStatus('PENDING', _PENDING), FakeStatus('SETUP_DONE', _SETUP_DONE)]
        job = _job('up-pts-f5014-pts_target-abcde', 'ERROR', _DONE, history)
        execution = job_execution(job, known_steps=['pts_target'])
        assert execution.started is None
        assert execution.execution_seconds is None

    def test_a_job_still_running_has_no_execution_time_yet(self) -> None:
        """The current status is `RUNNING` itself, not a terminal state.

        `status.state_start_time` here is when the job entered `RUNNING` -- the same
        instant as the `status_history` entry, not a later "it stopped" instant. If
        `ended` were taken unconditionally from `status.state_start_time`, this would
        report `execution_seconds == 0` for a job that has been running for however
        long it has, rather than reporting it as not yet knowable.
        """
        history = [
            FakeStatus('PENDING', _PENDING),
            FakeStatus('SETUP_DONE', _SETUP_DONE),
            FakeStatus('RUNNING', _RUNNING),
        ]
        job = _job('up-pts-f5014-pts_target-abcde', 'RUNNING', _RUNNING, history)
        execution = job_execution(job, known_steps=['pts_target'])
        assert execution.started == _RUNNING
        assert execution.ended is None
        assert execution.execution_seconds is None

    def test_a_cancelled_job_is_distinguishable_from_a_successful_one(self) -> None:
        done = job_execution(_finished_job('up-pts-f5014-pts_drug_molecule-c516h', 'DONE'))
        cancelled = job_execution(_finished_job('up-pts-f5014-pts_drug_molecule-gttbk', 'CANCELLED'))
        assert done.state == 'DONE'
        assert cancelled.state == 'CANCELLED'
        assert done.state != cancelled.state

    def test_the_state_name_is_read_off_a_real_shaped_enum_not_stringified(self) -> None:
        """Guards `_state_name` against a mutant that reads `str(state)` instead of `.name`.

        `str(_RealShapedState.DONE)` is `'_RealShapedState.DONE'`, not `'DONE'` -- a
        plain-string test double could not catch this, since a string's own `str()`
        is itself.
        """
        job = _finished_job('up-pts-f5014-pts_drug_molecule-c516h', _RealShapedState.DONE)
        job.status_history = [
            FakeStatus(_RealShapedState.PENDING, _PENDING),
            FakeStatus(_RealShapedState.SETUP_DONE, _SETUP_DONE),
            FakeStatus(_RealShapedState.RUNNING, _RUNNING),
        ]
        execution = job_execution(job, known_steps=['pts_drug_molecule'])
        assert execution.state == 'DONE'
        assert execution.started == _RUNNING


class TestStepForJobId:
    def test_the_step_is_recovered_despite_a_varying_job_id_prefix(self) -> None:
        """Two real prefix shapes, verified live: a short one and a longer, step-name-like one."""
        known_steps = ['pts_drug_molecule', 'pts_literature_ontoma']
        assert step_for_job_id('up-pts-f5014-pts_drug_molecule-c516h', known_steps) == 'pts_drug_molecule'
        assert (
            step_for_job_id('up-pts-literature-f5014-pts_literature_ontoma-5znfv', known_steps)
            == 'pts_literature_ontoma'
        )

    def test_a_job_id_matching_no_known_step_returns_none(self) -> None:
        assert step_for_job_id('up-something-unrelated-abcde', ['pts_drug_molecule']) is None

    def test_the_longest_matching_step_name_wins_regardless_of_list_order(self) -> None:
        """`pts_disease` is a substring of `pts_disease_hpo`; the more specific match must win.

        Checked in both orderings of `known_steps`, so a `next(...)`-style
        first-match implementation (correct only when the longer name happens to
        come first) cannot pass by accident.
        """
        job_id = 'up-pts-f5014-pts_disease_hpo-c516h'
        assert step_for_job_id(job_id, ['pts_disease', 'pts_disease_hpo']) == 'pts_disease_hpo'
        assert step_for_job_id(job_id, ['pts_disease_hpo', 'pts_disease']) == 'pts_disease_hpo'


class TestStepLabelIsAuthoritative:
    """F2: the job's own `step` label is read first; id-matching is only a fallback.

    Every case here sets the label and the id-matched step to *disagree* -- the shape
    verified live on `up-20260527-1458`, where `pts_target_safety`'s job ids matched
    the shorter, unrelated `pts_target` by substring while their labels correctly said
    `pts_target_safety`. A fixture where the two agree cannot tell "the label is
    preferred" apart from "id-matching happened to be right too".
    """

    def test_the_label_wins_over_a_disagreeing_id_match(self) -> None:
        job = _finished_job('up-pts-5df4f-pts_target-3omgp', labels={'step': 'pts_target_safety'})
        execution = job_execution(job, known_steps=['pts_target', 'pts_target_safety'])
        assert execution.step == 'pts_target_safety'

    def test_the_label_wins_even_when_the_id_matches_nothing_at_all(self) -> None:
        """The label alone is enough; id-matching does not even need to be attempted."""
        job = _finished_job('up-something-unrelated-abcde', labels={'step': 'pts_ontoma_literature'})
        execution = job_execution(job, known_steps=['pts_ontoma'])
        assert execution.step == 'pts_ontoma_literature'

    def test_an_absent_label_falls_back_to_id_matching(self) -> None:
        """No `step` key at all -- the shape every pre-F2 test in this module already covers."""
        job = _finished_job('up-pts-f5014-pts_drug_molecule-c516h', labels={})
        execution = job_execution(job, known_steps=['pts_drug_molecule'])
        assert execution.step == 'pts_drug_molecule'

    def test_an_empty_label_value_falls_back_to_id_matching(self) -> None:
        """A `step` label present but empty must not be treated as a real answer."""
        job = _finished_job('up-pts-f5014-pts_drug_molecule-c516h', labels={'step': ''})
        execution = job_execution(job, known_steps=['pts_drug_molecule'])
        assert execution.step == 'pts_drug_molecule'

    def test_a_label_naming_an_unknown_step_is_still_trusted(self) -> None:
        """The label is authoritative even when `known_steps` has renamed it away.

        `known_steps` here stands for the local checkout's yaml -- the `etl_literature`
        shape verified live, where the id-matching fallback alone found nothing at all.
        """
        job = _finished_job('up-etl-literature-5df4f-etl_literature-wz1ct', labels={'step': 'etl_literature'})
        execution = job_execution(job, known_steps=['pts_target'])
        assert execution.step == 'etl_literature'


class TestJobExecutionReportsUnmatchedSteps:
    def test_a_job_whose_step_matches_nothing_known_is_reported_with_step_none(self) -> None:
        """The job must still come back, distinct from `None`/being silently dropped."""
        job = _finished_job('up-something-unrelated-abcde')
        execution = job_execution(job, known_steps=['pts_drug_molecule'])
        assert execution.step is None
        assert execution.job_id == 'up-something-unrelated-abcde'


class TestJobExecutions:
    def test_every_job_the_client_returns_is_reported(self) -> None:
        """Guards against filtering the result down to only the matched-step jobs."""
        jobs = [_finished_job('up-pts-f5014-pts_drug_molecule-c516h'), _finished_job('up-something-unrelated-abcde')]
        client = FakeClient(jobs)
        executions = job_executions(client, 'proj', 'europe-west1', 'a-run', known_steps=['pts_drug_molecule'])
        assert {e.job_id for e in executions} == {
            'up-pts-f5014-pts_drug_molecule-c516h',
            'up-something-unrelated-abcde',
        }
        assert {e.step for e in executions} == {'pts_drug_molecule', None}

    def test_several_jobs_for_one_step_are_all_reported_not_collapsed(self) -> None:
        """The verified `pts_drug_molecule` shape: a cancelled re-run beside the job that succeeded.

        `job_executions` must not sum or dedupe these onto a single entry -- that
        choice belongs to `compute.py`'s join, not this layer. Two distinct
        `JobExecution`s, one per state, is the assertion.
        """
        jobs = [
            _finished_job('up-pts-f5014-pts_drug_molecule-c516h', 'DONE'),
            _finished_job('up-pts-f5014-pts_drug_molecule-gttbk', 'CANCELLED'),
        ]
        client = FakeClient(jobs)
        executions = job_executions(client, 'proj', 'europe-west1', 'a-run', known_steps=['pts_drug_molecule'])
        assert len(executions) == 2
        assert {e.state for e in executions} == {'DONE', 'CANCELLED'}
        assert all(e.step == 'pts_drug_molecule' for e in executions)

    def test_the_client_is_called_with_a_run_label_filter(self) -> None:
        client = FakeClient([])
        job_executions(client, 'my-project', 'europe-west1', 'manual__2026-07-21t15-07-47-545737-00-00')
        assert client.calls == [
            {
                'project_id': 'my-project',
                'region': 'europe-west1',
                'filter': 'labels.run = manual__2026-07-21t15-07-47-545737-00-00',
            }
        ]

    def test_an_empty_result_is_reported_as_an_empty_list(self) -> None:
        """Not asserted via a filtered comprehension that would also pass on a wrong-shaped result."""
        client = FakeClient([])
        executions = job_executions(client, 'proj', 'europe-west1', 'a-run')
        assert executions == []
        assert isinstance(executions, list)


class TestJobExecutionModel:
    def test_a_step_and_execution_seconds_can_both_be_none_at_once(self) -> None:
        """The model itself must not force a 0.0 default onto an unset execution time."""
        execution = JobExecution(job_id='j', step=None, state='PENDING')
        assert execution.execution_seconds is None
        assert execution.started is None
        assert execution.ended is None
