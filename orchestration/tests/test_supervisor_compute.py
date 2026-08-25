"""Tests for joining billed cost, Dataproc execution time and task wall time per step."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from orchestration.supervisor.compute import StepCompute, cluster_compute_report, compute_report
from orchestration.supervisor.dataproc import JobExecution
from orchestration.supervisor.journal import JournalEvent
from orchestration.supervisor.usage import StepUsage

_AT = datetime(2026, 7, 21, 17, 0, tzinfo=UTC)


def _usage(
    step: str,
    *,
    started: datetime = datetime(2026, 7, 21, 15, 0, tzinfo=UTC),
    ended: datetime = datetime(2026, 7, 21, 19, 0, tzinfo=UTC),
    billed_hours: int = 4,
    billed_hour_buckets: list[datetime] | None = None,
    net_cost: float = 6.69,
    currency: str = 'GBP',
    core_seconds: float | None = 583552.0,
    spot_core_seconds: float | None = 212831.0,
    machine_families: list[str] | None = None,
    cluster_instances: list[str] | None = None,
) -> StepUsage:
    return StepUsage(
        run='a-run',
        step=step,
        tool='pts',
        product='platform',
        started=started,
        ended=ended,
        billed_hours=billed_hours,
        billed_hour_buckets=(
            billed_hour_buckets
            if billed_hour_buckets is not None
            else [datetime(2026, 7, 21, 15 + h, 0, tzinfo=UTC) for h in range(billed_hours)]
        ),
        net_cost=net_cost,
        currency=currency,
        shared_cluster=False,
        core_seconds=core_seconds,
        spot_core_seconds=spot_core_seconds,
        machine_families=machine_families if machine_families is not None else ['N1'],
        cluster_instances=cluster_instances if cluster_instances is not None else [],
    )


def _execution(
    step: str | None,
    *,
    job_id: str = 'up-pts-f5014-x-c516h',
    state: str = 'DONE',
    execution_seconds: float | None = 1162.0,
    cluster_instance: str | None = None,
) -> JobExecution:
    return JobExecution(
        job_id=job_id, step=step, state=state, execution_seconds=execution_seconds, cluster_instance=cluster_instance
    )


def _completed(step: str, duration: float, *, try_number: int | None = 1) -> JournalEvent:
    return JournalEvent(
        event_type='step_completed', step=step, try_number=try_number, at=_AT, payload={'duration': duration}
    )


class TestAllThreeSources:
    def test_a_step_in_all_three_sources_gets_both_gaps(self) -> None:
        """The `pts_drug_molecule` shape: billed, executed, and journalled."""
        usages = [_usage('pts_drug_molecule', billed_hours=1)]
        executions = [_execution('pts_drug_molecule', execution_seconds=1162.0)]
        events = [_completed('pts_drug_molecule', 1191.0)]

        [step] = compute_report(usages, executions, events)

        assert step.step == 'pts_drug_molecule'
        assert step.execution_seconds == 1162.0
        assert step.wall_seconds == 1191.0
        assert step.billed_seconds == 3600.0
        assert step.billed_execution_gap_seconds == pytest.approx(3600.0 - 1162.0)
        assert step.wall_execution_gap_seconds == pytest.approx(1191.0 - 1162.0)


class TestNoExecutionTime:
    def test_a_step_with_no_dataproc_job_gets_neither_gap_and_says_so(self) -> None:
        """The common case: a GCE step, billed and journalled but never run on Dataproc."""
        usages = [_usage('pis_disease')]
        events = [_completed('pis_disease', 500.0)]

        [step] = compute_report(usages, [], events)

        assert step.execution_seconds is None
        assert step.dataproc_job_states == []
        assert step.billed_execution_gap_seconds is None
        assert step.wall_execution_gap_seconds is None
        assert step.wall_seconds == 500.0

    def test_a_step_whose_only_dataproc_job_was_cancelled_is_distinguishable_from_no_job_at_all(self) -> None:
        """Execution time is still absent, but `dataproc_job_states` says a job did run."""
        usages = [_usage('pts_drug_molecule')]
        executions = [_execution('pts_drug_molecule', state='CANCELLED', execution_seconds=None)]

        [step] = compute_report(usages, executions, [])

        assert step.execution_seconds is None
        assert step.dataproc_job_states == ['CANCELLED']
        assert step.billed_execution_gap_seconds is None


class TestBilledButNeverJournalled:
    def test_a_step_billed_but_never_journalled_still_appears(self) -> None:
        usages = [_usage('pts_reactome', billed_hours=4, net_cost=6.69)]

        [step] = compute_report(usages, [], [])

        assert step.step == 'pts_reactome'
        assert step.net_cost == 6.69
        assert step.billed_hours == 4
        assert step.wall_seconds is None
        assert step.wall_execution_gap_seconds is None


class TestSeveralJobsForOneStep:
    def test_a_cancelled_job_beside_a_done_job_does_not_inflate_execution_time(self) -> None:
        """The verified `pts_drug_molecule` shape: only the `DONE` job's time counts.

        Summing the `CANCELLED` job's execution time in would both inflate
        `execution_seconds` with thrown-away work and shrink `billed_execution_gap_seconds`,
        hiding the very waste that gap exists to show.
        """
        executions = [
            _execution('pts_drug_molecule', job_id='...-c516h', state='DONE', execution_seconds=1162.0),
            _execution('pts_drug_molecule', job_id='...-gttbk', state='CANCELLED', execution_seconds=678.0),
        ]

        [step] = compute_report([], executions, [])

        assert step.execution_seconds == 1162.0
        assert step.dataproc_job_states == ['CANCELLED', 'DONE']

    def test_two_done_jobs_for_one_step_are_summed(self) -> None:
        """Two genuinely successful jobs both did real, kept compute work."""
        executions = [
            _execution('pts_target', job_id='...-aaaaa', state='DONE', execution_seconds=100.0),
            _execution('pts_target', job_id='...-bbbbb', state='DONE', execution_seconds=50.0),
        ]

        [step] = compute_report([], executions, [])

        assert step.execution_seconds == 150.0

    def test_an_unmatched_job_id_is_not_joined_to_any_step(self) -> None:
        executions = [_execution(None, job_id='up-something-unrelated-abcde')]

        report = compute_report([], executions, [])

        assert report == []


class TestSharedClusterDetection:
    """F1: only Dataproc's own per-job data, never billing, can see a shared instance.

    A cluster instance's billing rows repeat only its creating step's label for the
    instance's whole life; only `cluster_instance` + `step` on `dataproc.JobExecution` can
    see a later step reusing it. A fixture where every cluster hosts exactly one step
    cannot exercise this at all -- see the class below for the shape that actually
    catches it.
    """

    def test_a_step_alone_on_its_own_cluster_instance_is_not_shared(self) -> None:
        usages = [_usage('pts_reactome', cluster_instances=['c1'], billed_hours=1)]
        executions = [_execution('pts_reactome', cluster_instance='c1', execution_seconds=100.0)]

        [step] = compute_report(usages, executions, [])

        assert step.shared_cluster is False
        assert step.billed_execution_gap_seconds == pytest.approx(3600.0 - 100.0)

    def test_another_step_s_job_on_the_same_instance_flags_the_billing_step_shared(self) -> None:
        """The verified `5b71ec48` shape, shrunk to two steps.

        Billing carries only `pts_reactome`'s label; Dataproc's job records show
        `pts_other` also ran a job on the same instance -- the fact billing alone can
        never see (see `usage.py`'s and `compute.py`'s module docstrings).
        """
        usages = [_usage('pts_reactome', cluster_instances=['c1'], billed_hours=1)]
        executions = [
            _execution('pts_reactome', job_id='j1', cluster_instance='c1', execution_seconds=100.0),
            _execution('pts_other', job_id='j2', cluster_instance='c1', execution_seconds=50.0),
        ]

        report = compute_report(usages, executions, [])
        reactome = next(s for s in report if s.step == 'pts_reactome')
        other = next(s for s in report if s.step == 'pts_other')

        assert reactome.shared_cluster is True
        assert reactome.billed_execution_gap_seconds is None, (
            "the billed 3600s is the instance's whole span, not pts_reactome's own"
        )
        assert reactome.execution_seconds == 100.0, "this step's own execution time is still reported"
        assert other.shared_cluster is False, 'pts_other billed nothing of its own to mis-flag'
        assert other.execution_seconds == 50.0

    def test_a_seventeen_step_cluster_still_flags_the_one_billing_step(self) -> None:
        """The real shape, at real scale: 17 distinct steps, one billing label.

        A fixture with only two steps sharing a cluster could pass by accident if the
        join only checked "exactly 2"; this checks the flag still fires, and still names
        only the billing step, when the count is much larger.
        """
        usages = [_usage('pts_reactome', cluster_instances=['c1'], billed_hours=4)]
        executions = [
            _execution(f'pts_step_{i}', job_id=f'j{i}', cluster_instance='c1', execution_seconds=10.0)
            for i in range(1, 17)
        ]
        executions.append(_execution('pts_reactome', job_id='j0', cluster_instance='c1', execution_seconds=153.8))

        report = compute_report(usages, executions, [])
        reactome = next(s for s in report if s.step == 'pts_reactome')

        assert reactome.shared_cluster is True
        assert reactome.billed_execution_gap_seconds is None
        assert sum(1 for s in report if s.shared_cluster) == 1, 'only the billing step is flagged, not all 17'

    def test_a_step_whose_usage_spans_two_clusters_also_gets_no_gap(self) -> None:
        """Ambiguous, so the step's own waste is not guessed at either.

        Folded into the same `shared_cluster` flag as an actually-shared instance, for
        the same reason: `billed_seconds` here would mix two physical instances' billed
        time into one number that is a duration on neither of them.
        """
        usages = [_usage('pts_reactome', cluster_instances=['c1', 'c2'], billed_hours=2)]
        executions = [
            _execution('pts_reactome', job_id='j1', cluster_instance='c1', execution_seconds=100.0),
            _execution('pts_other', job_id='j2', cluster_instance='c1', execution_seconds=50.0),
        ]

        step = next(s for s in compute_report(usages, executions, []) if s.step == 'pts_reactome')

        assert step.shared_cluster is True
        assert step.billed_execution_gap_seconds is None
        assert step.execution_seconds == 100.0, "this step's own execution time is still reported"

    def test_a_step_on_two_clusters_neither_individually_shared_still_gets_no_gap(self) -> None:
        """Neither `c1` nor `c2` is reused -- the ambiguity is only which one this billed time belongs to."""
        usages = [_usage('pts_reactome', cluster_instances=['c1', 'c2'], billed_hours=2)]
        executions = [_execution('pts_reactome', job_id='j1', cluster_instance='c1', execution_seconds=100.0)]

        [step] = compute_report(usages, executions, [])

        assert step.shared_cluster is True
        assert step.billed_execution_gap_seconds is None


class TestClusterComputeReport:
    """`cluster_compute_report`: the true idle time `StepCompute` cannot show once shared."""

    def test_an_unshared_instance_is_still_reported(self) -> None:
        usages = [_usage('pts_reactome', cluster_instances=['c1'], billed_hours=1)]
        executions = [_execution('pts_reactome', cluster_instance='c1', execution_seconds=100.0)]

        [cluster] = cluster_compute_report(usages, executions)

        assert cluster.cluster_instance == 'c1'
        assert cluster.billing_step == 'pts_reactome'
        assert cluster.steps == ['pts_reactome']
        assert cluster.shared is False
        assert cluster.billed_seconds == 3600.0
        assert cluster.execution_seconds == 100.0
        assert cluster.idle_seconds == pytest.approx(3500.0)

    def test_a_shared_instance_pools_every_step_s_execution_time(self) -> None:
        usages = [_usage('pts_reactome', cluster_instances=['c1'], billed_hours=1)]
        executions = [
            _execution('pts_reactome', job_id='j1', cluster_instance='c1', execution_seconds=100.0),
            _execution('pts_other', job_id='j2', cluster_instance='c1', execution_seconds=50.0),
        ]

        [cluster] = cluster_compute_report(usages, executions)

        assert cluster.shared is True
        assert cluster.steps == ['pts_other', 'pts_reactome']
        assert cluster.execution_seconds == 150.0, 'pooled across both steps, not just the billing step'
        assert cluster.idle_seconds == pytest.approx(3600.0 - 150.0)

    def test_a_cancelled_job_on_a_shared_instance_does_not_inflate_the_pooled_execution_time(self) -> None:
        usages = [_usage('pts_reactome', cluster_instances=['c1'], billed_hours=1)]
        executions = [
            _execution('pts_reactome', job_id='j1', cluster_instance='c1', execution_seconds=100.0),
            _execution('pts_other', job_id='j2', cluster_instance='c1', state='CANCELLED', execution_seconds=678.0),
        ]

        [cluster] = cluster_compute_report(usages, executions)

        assert cluster.steps == ['pts_other', 'pts_reactome'], 'still counted as having used the instance'
        assert cluster.execution_seconds == 100.0, 'the cancelled job did not reach DONE'

    def test_an_instance_with_a_job_but_no_attributable_billing_has_no_billed_seconds(self) -> None:
        """A cluster billing hasn't caught up to yet, or whose billing step is ambiguous."""
        executions = [_execution('pts_reactome', cluster_instance='c1', execution_seconds=100.0)]

        [cluster] = cluster_compute_report([], executions)

        assert cluster.billing_step is None
        assert cluster.billed_seconds is None
        assert cluster.currency is None
        assert cluster.idle_seconds is None
        assert cluster.execution_seconds == 100.0

    def test_a_gce_only_run_reports_no_clusters(self) -> None:
        usages = [_usage('pis_disease', cluster_instances=[])]

        assert cluster_compute_report(usages, []) == []

    def test_clusters_are_sorted_by_instance_uuid(self) -> None:
        usages = [
            _usage('pts_zzz', cluster_instances=['zzz-uuid'], billed_hours=1),
            _usage('pts_aaa', cluster_instances=['aaa-uuid'], billed_hours=1),
        ]

        clusters = cluster_compute_report(usages, [])

        assert [c.cluster_instance for c in clusters] == ['aaa-uuid', 'zzz-uuid']


class TestOrdering:
    def test_the_report_is_sorted_by_step_name_regardless_of_input_order(self) -> None:
        usages = [_usage('pts_zzz'), _usage('pts_aaa')]
        executions = [_execution('pts_mmm')]
        events = [_completed('pts_bbb', 10.0)]

        report = compute_report(usages, executions, events)

        assert [s.step for s in report] == ['pts_aaa', 'pts_bbb', 'pts_mmm', 'pts_zzz']


class TestMissingSourcesDoNotRenderAsZero:
    def test_a_step_with_no_usage_row_has_no_cost_fields_at_all(self) -> None:
        """Guards against a mutant that defaults a missing usage row's cost to `0.0`.

        A `0.0` net cost claims "billed and it cost nothing", which is a different and
        false claim from "no usage row was ever seen for this step".
        """
        [step] = compute_report([], [_execution('pts_target')], [])

        assert step.net_cost is None
        assert step.currency is None
        assert step.billed_hours is None
        assert step.billed_seconds is None
        assert step.core_hours is None
        assert step.spot_share is None

    def test_a_step_present_only_in_the_journal_still_appears(self) -> None:
        [step] = compute_report([], [], [_completed('pts_only_in_journal', 42.0)])

        assert step.step == 'pts_only_in_journal'
        assert step.wall_seconds == 42.0
        assert step.net_cost is None
        assert step.execution_seconds is None

    def test_core_seconds_with_no_spot_billed_is_a_measured_zero_not_none(self) -> None:
        usages = [_usage('pts_target', core_seconds=1000.0, spot_core_seconds=0.0)]

        [step] = compute_report(usages, [], [])

        assert step.spot_core_seconds == 0.0
        assert step.spot_share == 0.0


class TestCurrencyMismatchRaises:
    def test_two_usage_rows_for_the_same_step_in_different_currencies_raise(self) -> None:
        """Silently summing across currencies would produce a number that looks like money and is not."""
        usages = [_usage('pts_target', currency='GBP'), _usage('pts_target', currency='USD')]

        with pytest.raises(ValueError, match='more than one currency'):
            compute_report(usages, [], [])


class TestMultipleUsageRowsForOneStep:
    def test_cost_and_core_seconds_are_summed_across_rows_for_the_same_step(self) -> None:
        usages = [
            _usage(
                'pts_target', net_cost=1.0, billed_hours=1, core_seconds=100.0, spot_core_seconds=0.0,
                billed_hour_buckets=[datetime(2026, 7, 21, 15, 0, tzinfo=UTC)],
            ),
            _usage(
                'pts_target', net_cost=2.5, billed_hours=2, core_seconds=200.0, spot_core_seconds=50.0,
                billed_hour_buckets=[
                    datetime(2026, 7, 21, 16, 0, tzinfo=UTC), datetime(2026, 7, 21, 17, 0, tzinfo=UTC),
                ],
            ),
        ]

        [step] = compute_report(usages, [], [])

        assert step.net_cost == pytest.approx(3.5)
        assert step.billed_hours == 3
        assert step.core_seconds == 300.0
        assert step.spot_core_seconds == 50.0


class TestBilledHoursIsAUnionNotASum:
    """F7: two `StepUsage` rows for one step (a `tool`/`product` split) can share an hour.

    Summing each row's own `billed_hours` would double-count that hour. Real case: a step
    billed under both `platform` and `ppp` in the same run, in the same clock hour.
    """

    def test_an_hour_both_rows_billed_in_is_not_counted_twice(self) -> None:
        """The two rows stand in for a `platform` row and a `ppp` row sharing an hour."""
        shared_hour = datetime(2026, 7, 21, 15, 0, tzinfo=UTC)
        usages = [
            _usage('pts_target', billed_hours=1, billed_hour_buckets=[shared_hour]),
            _usage(
                'pts_target', billed_hours=2,
                billed_hour_buckets=[shared_hour, datetime(2026, 7, 21, 16, 0, tzinfo=UTC)],
            ),
        ]

        [step] = compute_report(usages, [], [])

        assert step.billed_hours == 2, 'summing the scalar counts (1 + 2) would wrongly give 3'
        assert step.billed_seconds == 7200.0

    def test_the_envelope_spans_the_earliest_start_and_latest_end_across_rows(self) -> None:
        usages = [
            _usage(
                'pts_target',
                started=datetime(2026, 7, 21, 15, 0, tzinfo=UTC),
                ended=datetime(2026, 7, 21, 16, 0, tzinfo=UTC),
            ),
            _usage(
                'pts_target',
                started=datetime(2026, 7, 21, 14, 0, tzinfo=UTC),
                ended=datetime(2026, 7, 21, 17, 0, tzinfo=UTC),
            ),
        ]

        [step] = compute_report(usages, [], [])

        assert step.billed_envelope_started == datetime(2026, 7, 21, 14, 0, tzinfo=UTC)
        assert step.billed_envelope_ended == datetime(2026, 7, 21, 17, 0, tzinfo=UTC)


class TestWallTimeTakesTheMaxAcrossRepeatedCompletions:
    def test_a_step_re_run_within_one_run_reports_the_larger_completion(self) -> None:
        """Matches `stall.baseline_from_journal`'s own tie-breaking rule, reused here."""
        events = [
            _completed('pts_clinical_report', 678.0, try_number=1),
            _completed('pts_clinical_report', 3094.0, try_number=2),
        ]

        [step] = compute_report([], [], events)

        assert step.wall_seconds == 3094.0

    def test_the_max_is_reported_even_when_the_larger_completion_is_journalled_first(self) -> None:
        """Distinguishes "take the max" from "take the last event seen".

        A dict keyed by step and overwritten in journal order would agree with the max
        whenever the later attempt happens to be the longer one, as in the test above --
        which is also the realistic case, since a retry after a stall is often slower, not
        faster. Reversing which try is longer is what actually forces the max, not merely
        the most recent value, to be the one computed.
        """
        events = [
            _completed('pts_clinical_report', 3094.0, try_number=1),
            _completed('pts_clinical_report', 678.0, try_number=2),
        ]

        [step] = compute_report([], [], events)

        assert step.wall_seconds == 3094.0


class TestStepComputeModel:
    def test_a_step_with_nothing_measured_at_all_is_still_constructible(self) -> None:
        step = StepCompute(step='pts_nothing_known')
        assert step.net_cost is None
        assert step.execution_seconds is None
        assert step.wall_seconds is None
        assert step.billed_execution_gap_seconds is None
        assert step.wall_execution_gap_seconds is None
        assert step.machine_families == []
        assert step.dataproc_job_states == []

    def test_a_negative_gap_is_reported_rather_than_clamped(self) -> None:
        """A step that executed longer than its billed span is a real, surfaceable anomaly.

        Clamping it to zero would hide a labelling or billing-window defect behind a
        number that looks like clean, unremarkable data.
        """
        step = StepCompute(step='pts_target', billed_hours=1, execution_seconds=4000.0)
        gap = step.billed_execution_gap_seconds
        assert gap == pytest.approx(3600.0 - 4000.0)
        assert gap is not None
        assert gap < 0
