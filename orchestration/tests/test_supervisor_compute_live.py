"""Live checks against the real billing export, Dataproc API and journal, joined.

Skipped unless RUN_COMPUTE_TESTS is set, because these need BigQuery, Dataproc and GCS
credentials plus network. Run with:
RUN_COMPUTE_TESTS=1 uv run --frozen pytest tests/test_supervisor_compute_live.py -rxs

Like `test_supervisor_usage_live.py` and `test_supervisor_dataproc_live.py`, this is a
documented manual procedure, not a CI guarantee: `.github/workflows/check.yaml` runs a
bare `pytest` with none of RUN_BIGQUERY_TESTS, RUN_DATAPROC_TESTS or RUN_COMPUTE_TESTS
set, so this module never runs there. Run it by hand before trusting a change to
`compute.py` or the `compute` CLI command against production.

Every figure this module pins was verified by hand against production on 2026-08-24,
against the run below. Numbers drift as the export gathers more days of data or the
Dataproc job history rolls off; the properties that must not drift -- absence never
reading as zero, a gap computed only when both its inputs are present, a cancelled job
staying distinct from no job at all, coverage always travelling with the report -- are
asserted structurally, not by re-deriving the pinned numbers.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC, date, datetime

import pytest
from google.cloud import bigquery, dataproc_v1, storage

from orchestration.supervisor.cli import render_compute, render_coverage
from orchestration.supervisor.compute import StepCompute, compute_report
from orchestration.supervisor.dataproc import JobExecution, job_executions
from orchestration.supervisor.journal import Journal
from orchestration.supervisor.usage import BillingExport, StepUsage, WindowCoverage, usage_window
from orchestration.utils.common import GCP_PROJECT_PLATFORM, GCP_REGION, GCS_PIPELINE_RUNS_BUCKET

pytestmark = pytest.mark.skipif(
    not os.environ.get('RUN_COMPUTE_TESTS'),
    reason='needs BigQuery, Dataproc and GCS credentials, set RUN_COMPUTE_TESTS=1 to run',
)

SINCE = date(2026, 5, 1)
"""Earliest partition date to scan -- the export holds nothing before this."""

RAW_RUN = 'manual__2026-07-21T15:07:47.545737+00:00'
"""The Airflow `dag_run_id`, unmodified -- what the journal is keyed on (see
`journal.py`'s module docstring), and what `cli.py`'s `compute` command reads
`--run` as before cleaning it for the billing/Dataproc queries below."""

KNOWN_RUN = 'manual__2026-07-21t15-07-47-545737-00-00'
"""`clean_label(RAW_RUN)` -- the same cleaned label `test_supervisor_usage_live.py`'s
and `test_supervisor_dataproc_live.py`'s `KNOWN_RUN` verify."""

REACTOME_STEP = 'pts_reactome'
"""Verified live 2026-08-24 (also see `test_supervisor_usage_live.py`'s
`REACTOME_STEP`): this run's most expensive step, and the one whose billed envelope
most exceeds what it actually computed -- 4 billed hours and 162.1 core-hours for
153.8s of Dataproc execution, costing 6.69 GBP net."""

REACTOME_JOB_ID = 'up-pts-f5014-pts_reactome-ym95s'
"""The one Dataproc job behind `REACTOME_STEP` in this run: RUNNING ~15:14:18 -> DONE
~15:16:52, verified against the raw Job Controller API response. Held to whole
seconds, generously toleranced below -- `status_history` timestamps are precise to
the microsecond and this project's own re-queries of the same job have landed a
fraction of a second apart, which is not itself a finding."""

REACTOME_STARTED_APPROX = datetime(2026, 7, 21, 15, 14, 18, tzinfo=UTC)
"""See `REACTOME_JOB_ID`."""

REACTOME_ENDED_APPROX = datetime(2026, 7, 21, 15, 16, 52, tzinfo=UTC)
"""See `REACTOME_JOB_ID`."""

EXPECTED_STEP_COUNT = 34
"""The run joins cost, execution time and journal events into this many step rows --
a union of every step any of the three sources mentions, not an intersection."""

EXPECTED_NO_JOB_COUNT = 15
"""Of `EXPECTED_STEP_COUNT`, this many have no Dataproc job at all (`[]`) -- the normal
shape for a step that runs on a plain GCE VM rather than Dataproc."""

EXPECTED_COVERAGE_SHARE = 0.697
"""The report covers 69.7% of the window's pipeline spend: 21.60 of 30.99 GBP."""

RERUN_STEPS_STATES = {
    'pts_drug_molecule': {'CANCELLED', 'DONE'},
    'pts_clinical_report': {'DONE', 'ERROR'},
}
"""Two steps whose jobs were re-run within this run, verified live 2026-08-24. Each
carries a job that did not reach DONE *beside* one that did -- the shape that must
never be miscounted as "no Dataproc job", the twelfth defect this project shipped
(see `compute.py`'s module docstring and `test_supervisor_dataproc_live.py`'s
`KNOWN_RUN_DRUG_MOLECULE_JOB_STATES`)."""


@dataclass
class LiveCompute:
    """Every source this module needs, fetched once and joined once.

    A BigQuery scan and a Dataproc list call both cost real quota, and every test in
    this module asks a question about the same run -- fetching per-test like
    `test_supervisor_usage_live.py` does would multiply that cost by the number of
    tests for no benefit, since none of them mutate what they read.
    """

    usages: list[StepUsage]
    executions: list[JobExecution]
    steps: list[StepCompute]
    window: tuple[datetime, datetime] | None
    coverage: list[WindowCoverage]
    rendered: str


@pytest.fixture(scope='module')
def live() -> LiveCompute:
    export = BillingExport(client=bigquery.Client(project=GCP_PROJECT_PLATFORM))
    usages = export.run_usage(run=KNOWN_RUN, since=SINCE)
    window = usage_window(usages)
    coverage = export.window_coverage(run=KNOWN_RUN, window=window, since=SINCE) if window is not None else []

    dataproc_client = dataproc_v1.JobControllerClient(
        client_options={'api_endpoint': f'{GCP_REGION}-dataproc.googleapis.com:443'}
    )
    executions = job_executions(dataproc_client, project=GCP_PROJECT_PLATFORM, region=GCP_REGION, run=KNOWN_RUN)

    bucket = storage.Client().bucket(GCS_PIPELINE_RUNS_BUCKET)
    journal = Journal(bucket=bucket, prefix=f'_agent/unified_pipeline/{RAW_RUN}/journal')
    events = journal.read()

    steps = compute_report(usages, executions, events)
    rendered = render_compute(steps) + '\n\n' + render_coverage(window, coverage)
    return LiveCompute(
        usages=usages, executions=executions, steps=steps, window=window, coverage=coverage, rendered=rendered
    )


class TestLiveJoinShape:
    """The join's own shape: how many steps, and how many of them have no Dataproc job.

    Both counts are asserted on exact contents, not on "the result is non-empty" --
    the failure this project has shipped repeatedly. An empty join, or one that
    silently dropped rows, would satisfy a non-emptiness check just as well as the
    real thing.
    """

    def test_the_run_joins_the_verified_step_count(self, live: LiveCompute) -> None:
        assert len(live.steps) == EXPECTED_STEP_COUNT

    def test_the_verified_count_of_steps_have_no_dataproc_job(self, live: LiveCompute) -> None:
        no_job = [s for s in live.steps if not s.dataproc_job_states]
        assert len(no_job) == EXPECTED_NO_JOB_COUNT


class TestAbsentIsNeverZero:
    """The property that matters more than any single figure here.

    For every step with no Dataproc job, execution time and both gaps must be
    `None`, never `0` or `0.0` -- a `0` would claim "measured, and there was no
    waste", a materially different and false claim from "this cannot be judged on
    that axis at all". See `compute.py`'s module docstring.
    """

    def test_a_step_with_no_dataproc_job_has_no_execution_time_or_gaps(self, live: LiveCompute) -> None:
        no_job = [s for s in live.steps if not s.dataproc_job_states]
        assert no_job, 'the fixture asserting EXPECTED_NO_JOB_COUNT already confirms this is non-empty'
        for step in no_job:
            assert step.execution_seconds is None
            assert step.billed_execution_gap_seconds is None
            assert step.wall_execution_gap_seconds is None

    def test_gaps_are_computed_only_when_both_inputs_are_present(self, live: LiveCompute) -> None:
        """Checked as a biconditional over every step, not spot-checked on one row.

        A join that dropped `billed_seconds` while keeping `execution_seconds` (or the
        reverse) would still pass a test that only checks the `None` direction.
        """
        for step in live.steps:
            has_billed_gap = step.billed_execution_gap_seconds is not None
            both_present = step.billed_seconds is not None and step.execution_seconds is not None
            assert has_billed_gap == both_present, step.step

            has_wall_gap = step.wall_execution_gap_seconds is not None
            both_present = step.wall_seconds is not None and step.execution_seconds is not None
            assert has_wall_gap == both_present, step.step


class TestNoJournalForThisRun:
    """This run predates the observer, so it has no journal -- verified, not assumed."""

    def test_the_journal_is_empty_for_this_run(self, live: LiveCompute) -> None:
        """Distinguishes "no journal exists" from "the journal read silently failed"."""
        assert all(step.wall_seconds is None for step in live.steps)

    def test_the_rendered_report_explains_the_absence_rather_than_showing_zeros(self, live: LiveCompute) -> None:
        assert 'wall' not in live.rendered.split('\n\n')[0].splitlines()[0]  # column hidden
        assert 'never taken' in live.rendered
        assert 'observer' in live.rendered


class TestReRunStepsKeepEveryJobState:
    """`dataproc_job_states` must not collapse a re-run into a single verdict.

    Both steps below carry a job that never reached DONE *beside* one that did.
    Miscounting either as "no Dataproc job" was the twelfth false guard on this
    project, caught only by mutation -- see `compute.py`'s module docstring.

    Neither step here is the case that guard actually keys on, though: both carry a
    DONE job, so `execution_seconds` happens to be set for them too, and keying the
    "no Dataproc job" count on `execution_seconds` instead of `dataproc_job_states`
    would not misclassify either one. No step in this run has a job that ran and
    never reached DONE with nothing else alongside it, so that exact edge stays
    unverifiable live; `test_supervisor_cli.py`'s
    `test_a_cancelled_only_job_does_not_count_as_no_dataproc_job` covers it
    synthetically instead. What this class confirms live is the shape one level up:
    a re-run really does leave every job's state on the record, not just the last
    or the "best" one.
    """

    @pytest.mark.parametrize(('step_name', 'expected_states'), list(RERUN_STEPS_STATES.items()))
    def test_the_re_run_steps_show_every_job_state(
        self, live: LiveCompute, step_name: str, expected_states: set
    ) -> None:
        step = next(s for s in live.steps if s.step == step_name)
        assert set(step.dataproc_job_states) == expected_states

    @pytest.mark.parametrize('step_name', list(RERUN_STEPS_STATES))
    def test_the_done_job_is_still_summed_despite_the_failed_one_alongside_it(
        self, live: LiveCompute, step_name: str
    ) -> None:
        """Only the DONE job's time counts -- see `compute.py`'s `_KEPT_JOB_STATE`.

        A join that instead summed every job regardless of state would inflate this
        step's execution time with CPU that produced nothing kept, and would shrink
        `billed_execution_gap_seconds` by exactly that amount -- hiding the waste
        the gap exists to surface.
        """
        step = next(s for s in live.steps if s.step == step_name)
        assert step.execution_seconds is not None
        done_jobs = [e for e in live.executions if e.step == step_name and e.state == 'DONE']
        assert len(done_jobs) == 1, 'the fixture data for these two steps is one DONE job each'
        assert step.execution_seconds == pytest.approx(done_jobs[0].execution_seconds, abs=0.01)


class TestReactomeMatchesTheVerifiedFigures:
    """`pts_reactome`, the run's most expensive step: pinned exactly.

    These are a measured snapshot from 2026-08-24, not a durable property -- a future
    failure here most likely means the export or Dataproc history moved, not that the
    join broke. `TestAbsentIsNeverZero` and `TestLiveJoinShape` above are what would
    actually catch the join returning garbage; this class is what confirms the numbers
    the report is *for*.
    """

    @pytest.fixture
    def reactome(self, live: LiveCompute) -> StepCompute:
        return next(s for s in live.steps if s.step == REACTOME_STEP)

    def test_cost_and_billed_hours(self, reactome: StepCompute) -> None:
        assert reactome.net_cost == pytest.approx(6.69, abs=0.01)
        assert reactome.billed_hours == 4

    def test_execution_time_and_waste(self, reactome: StepCompute) -> None:
        assert reactome.execution_seconds == pytest.approx(153.8, abs=1.0)
        assert reactome.billed_execution_gap_seconds == pytest.approx(14246.2, abs=2.0)

    def test_core_hours_and_spot_share(self, reactome: StepCompute) -> None:
        assert reactome.core_hours == pytest.approx(162.1, abs=0.1)
        assert reactome.spot_share == pytest.approx(0.36, abs=0.02)
        assert reactome.machine_families == ['N1']

    def test_its_one_dataproc_job_matches_the_raw_api_record(self, live: LiveCompute) -> None:
        """Cross-checks `compute_report`'s aggregate against the un-joined execution record.

        `StepCompute` drops job ids, so this is the only place in the module that
        confirms the number above actually comes from the job the ledger names.
        """
        execution = next(e for e in live.executions if e.job_id == REACTOME_JOB_ID)
        assert execution.step == REACTOME_STEP
        assert execution.state == 'DONE'
        assert execution.started is not None
        assert execution.ended is not None
        assert abs((execution.started - REACTOME_STARTED_APPROX).total_seconds()) < 2
        assert abs((execution.ended - REACTOME_ENDED_APPROX).total_seconds()) < 2
        assert execution.execution_seconds == pytest.approx(153.8, abs=1.0)


class TestCoverage:
    """Coverage travels with the report -- a total is never mistaken for the whole run."""

    def test_the_window_is_found(self, live: LiveCompute) -> None:
        assert live.window is not None

    def test_coverage_matches_the_verified_share(self, live: LiveCompute) -> None:
        assert live.coverage
        entry = live.coverage[0]
        assert entry.currency == 'GBP'
        assert entry.labelled_cost == pytest.approx(21.60, abs=0.05)
        assert entry.pipeline_cost == pytest.approx(30.99, abs=0.05)
        assert entry.labelled_share == pytest.approx(EXPECTED_COVERAGE_SHARE, abs=0.005)

    def test_labelled_spend_never_exceeds_pipeline_spend(self, live: LiveCompute) -> None:
        entry = live.coverage[0]
        assert entry.labelled_cost < entry.pipeline_cost
        assert not entry.exceeds_pipeline_cost

    def test_the_rendered_report_states_the_coverage_next_to_the_table(self, live: LiveCompute) -> None:
        """The durable form of the pinned figure above.

        The coverage section must exist, name a real share, and sit beside the
        per-step table -- not only live in a value a caller could discard.
        """
        table, _, coverage_section = live.rendered.partition('\n\ncoverage:')
        assert table.strip(), 'the per-step table must render before the coverage note'
        assert coverage_section, 'the coverage note must be present, not folded away'
        assert 'GBP' in coverage_section
        assert '%' in coverage_section
