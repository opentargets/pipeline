"""Join billed cost, Dataproc execution time and Airflow task wall time into one per-step picture.

Three sources, three different questions, none of them redundant with either other:

- `usage.StepUsage` — what the step **cost**, and the hourly buckets it was billed in.
  Hour-quantised: it cannot tell duration apart from an envelope that outlived the step
  (see `usage.py`'s module docstring).
- `dataproc.JobExecution` — how long a Dataproc job actually spent **computing**, from the
  Job Controller API's own state history. Only exists for the subset of steps that run a
  pyspark job on Dataproc; every other step runs on a plain GCE VM and has none.
- The journal's `step_completed` events — how long the step's own Airflow task took, start
  to finish (`TaskInstance.duration`; see `stall.py`'s module docstring for why that
  excludes scheduler queueing but still includes Dataproc job submission, polling and
  teardown around the job's own execution window). This is what the rest of this module
  calls **task wall time**.

**The two gaps are the point.** `billed - execution` is cluster time paid for and not
computing — the single most actionable number here. `wall - execution` is queueing under
contention: a step whose task wall time is much larger than its execution time is not slow
to compute, it is slow to get a turn on a busy cluster, and that calls for cluster sizing,
not a faster transform. Both are computed only when both of their inputs are present, and
`StepCompute` represents "not computable" as `None`, never as `0.0` — a step with no
Dataproc job is not a step with no waste, it is a step this report cannot judge on that
axis at all.

**Task wall time reuses `stall.baseline_from_journal` rather than re-deriving it.** That
function already turns a run's `step_completed` events into "the observed maximum
`duration` per step, keyed by the bare step name" — exactly what this module needs, and
already exercised by `stall.py`'s own tests. The one caveat carries over unchanged: if a
step's own run task completed more than once in the same run (cleared and re-run), the
figure reported here is the larger of the two, the same choice `stall.py` makes for its
baseline. Re-deriving the same max-per-step scan here would only give the same answer a
second, divergence-prone way.

**Several Dataproc jobs for one step is the case `dataproc.py` deliberately left
unresolved**, and it is not theoretical: a verified run shows `pts_drug_molecule` with a
`CANCELLED` job beside the `DONE` one that actually succeeded — an observed re-run, not a
hypothetical. This module's rule: **only `DONE` jobs contribute to a step's
`execution_seconds`, summed if there is more than one.** A `CANCELLED` or `ERROR` job spent
real CPU that the billing export still charges the step for, but produced nothing the step
kept, so folding it into "how long the step took to compute" would inflate that figure with
work that was thrown away — and would also *shrink* `billed - execution`, hiding exactly the
waste that gap exists to show. Every job's state is still kept, in `dataproc_job_states`
(one entry per job, states repeated if more than one job shares a state), so a reader can
tell "no Dataproc job ran" (`[]`) apart from "job(s) ran but none reached `DONE`" (non-empty,
`execution_seconds` still `None`) apart from "one or more jobs finished, and this is their
summed compute time" (non-empty, `execution_seconds` set) — the same absent-vs-zero
distinction this module applies everywhere else, applied to *why* a value is absent too.

Steps are joined by their bare `unified_pipeline.yaml` name, the one spelling all three
sources already agree on: `usage.StepUsage.step` is the billing `step` label,
`dataproc.JobExecution.step` is recovered from the job id against the same step list, and a
journalled `step_completed` event's `step` is validated by `JournalEvent` itself to be the
bare name, never a qualified Airflow `task_id`. The result is the union of every step any of
the three sources mentions, so a step billed but never journalled, or journalled with no
Dataproc job, still gets a row rather than being silently dropped by whichever source is
treated as primary.

**`billed - execution` is only the step's own waste when the step's billed cluster instance
served no other step.** `operators/dataproc.py` submits with `use_if_exists=True`: a cluster
instance created by one step can go on to run jobs submitted by several later steps, while
its billing rows -- stamped once, at creation -- name only the step that created it (see
`usage.py`'s module docstring). Subtracting that one step's `execution_seconds` from the
*whole instance's* billed time, as if the instance served only that step, both overstates the
creator's own waste and hides every other step's real usage of the same money. Verified live
2026-08-24 (F1 in this project's review ledger): cluster instance `5b71ec48` billed
9,792s of Dataproc execution across 17 distinct steps in one run, entirely under
`step=pts_reactome`'s label -- naively computed, `pts_reactome`'s own row would have claimed
essentially the whole instance's billed span as its own idle time, when 16 other steps'
jobs accounted for the rest of what actually ran there.

This module's answer: `StepCompute.shared_cluster` is set, from `dataproc.JobExecution`'s own
`cluster_instance` and `step` fields (never from billing, which cannot see this -- see
`usage.py`), whenever more than one step submitted a job to the instance a step's usage is
billed against. A shared step's `billed_execution_gap_seconds` reads `None`, not a wrong
number: the instance's true idle time is still measurable, just not attributable to one step,
and `cluster_compute_report` is where it is reported instead, pooled at the instance level
across every step that used it.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime

from pydantic import BaseModel, Field, computed_field

from orchestration.supervisor.dataproc import JobExecution
from orchestration.supervisor.journal import JournalEvent
from orchestration.supervisor.stall import baseline_from_journal
from orchestration.supervisor.usage import StepUsage


def unresolved_job_count(executions: Iterable[JobExecution]) -> int:
    """Count Dataproc jobs whose step could not be resolved to any known step name.

    `compute_report` skips exactly these -- an execution with `step is None` cannot be
    joined to any step's row, so it contributes to no `StepCompute` (see
    `compute_report`'s `executions` arg). `dataproc.JobExecution.step`'s own docstring
    promises an unresolved job is "reported rather than dropped", and `job_executions`
    keeps that promise at its own layer -- but a caller that only looks at
    `compute_report`'s result would still see it vanish with nothing to show anything
    was missing. This function is what lets a caller (`cli.py`'s `compute` command)
    keep that promise at the assembled level too: call it on the same `executions`
    passed to `compute_report`, and surface the count alongside the per-step table
    rather than discarding it silently.

    Args:
        executions: Job executions, normally the same list passed to `compute_report`.

    Returns:
        How many executions have `step is None`.
    """
    return sum(1 for execution in executions if execution.step is None)


_KEPT_JOB_STATE = 'DONE'
"""The only `JobExecution.state` that contributes to a step's `execution_seconds`.

See this module's docstring for why `CANCELLED` and `ERROR` are excluded rather than
summed in: they used CPU the billing export still charges for, but the gap that CPU time
opens against the billed span is exactly the waste `billed_execution_gap_seconds` exists
to surface, not something to hide by folding it into "how long the step took"."""


class StepCompute(BaseModel):
    """One step's cost, compute time and task wall time, joined from three sources.

    Every field below that can legitimately be absent for a step defaults to `None` (or
    `[]` for a list) rather than a `0`-like value — see this module's docstring. A `None`
    on any of `net_cost`, `billed_hours` etc. means the step had no `usage.StepUsage` row
    at all; a `None` on `execution_seconds` means it had no Dataproc job, or none of its
    jobs reached `DONE` — `dataproc_job_states` tells the two apart; a `None` on
    `wall_seconds` means the journal held no `step_completed` event for it.

    Args:
        step: The bare `unified_pipeline.yaml` step name, e.g. `pts_target`.
        net_cost: Summed `StepUsage.net_cost` across every usage row for this step.
            `None` if the step billed nothing under a `step` label in this report's input.
        currency: The currency `net_cost` is denominated in. `None` under exactly the
            condition `net_cost` is `None`.
        billed_hours: The number of distinct hourly buckets billed across this step's
            usage rows, not an envelope and not a sum of each row's own `billed_hours` —
            summing would double-count an hour two rows both billed in (a step billed
            under both `platform` and `ppp` in the same run, in the same hour, is the
            real case this guards: see `usage.StepUsage.billed_hour_buckets`). `None`
            under the same condition as `net_cost`.
        billed_envelope_started: Earliest `StepUsage.started` across this step's usage
            rows. An envelope, not a duration — see `billed_seconds` and `usage.py`'s
            module docstring before treating `billed_envelope_ended -
            billed_envelope_started` as how long the step took.
        billed_envelope_ended: Latest `StepUsage.ended` across this step's usage rows.
            Same caveat as `billed_envelope_started`.
        core_seconds: Summed `StepUsage.core_seconds` across this step's usage rows that
            billed a core SKU at all. `None` if none did — a Batch step, or one whose cost
            is storage/network only.
        spot_core_seconds: The subset of `core_seconds` billed under a spot SKU. `None`
            under exactly the condition `core_seconds` is `None`.
        machine_families: Distinct machine families across this step's usage rows,
            sorted. Empty when `core_seconds` is `None`, for the same reason.
        execution_seconds: Summed `JobExecution.execution_seconds` across this step's
            `DONE` Dataproc jobs only — see this module's docstring. `None` if the step had
            no Dataproc job, or none of its jobs reached `DONE`.
        dataproc_job_states: Every `JobExecution.state` seen for this step's jobs, one
            entry per job (states repeat if more than one job shares one), sorted. Empty
            means no Dataproc job at all for this step — the common case, since only
            pyspark steps run on Dataproc.
        wall_seconds: The step's own Airflow run task duration, from the journal, via
            `stall.baseline_from_journal` — see this module's docstring for what that
            duration does and does not include, and for the max-of-several-completions
            rule. `None` if the journal held no `step_completed` event for this step.
        shared_cluster: True when this step's own `billed_execution_gap_seconds` cannot be
            trusted as this step's waste, for either of two reasons: (1) the step's billed
            usage is attributable to exactly one Dataproc cluster instance, and that
            instance also ran a Dataproc job submitted by at least one other step (per
            `dataproc.JobExecution.cluster_instance` and `.step` — never from billing,
            which cannot see this; see this module's and `usage.py`'s docstrings), or (2)
            the step's usage spans more than one cluster instance, so `billed_seconds`
            would mix two physical instances' billed time into one number that is not a
            duration on either of them. Either way `billed_execution_gap_seconds` reads
            `None`, and case (1)'s true idle time is still measurable — see
            `cluster_compute_report` — pooled at the instance level across every step that
            used it. False, never `None`, for a step confirmed to have an instance to
            itself, and for a step with no Dataproc cluster billing at all.
    """

    step: str
    net_cost: float | None = None
    currency: str | None = None
    billed_hours: int | None = None
    billed_envelope_started: datetime | None = None
    billed_envelope_ended: datetime | None = None
    core_seconds: float | None = None
    spot_core_seconds: float | None = None
    machine_families: list[str] = Field(default_factory=list)
    execution_seconds: float | None = None
    dataproc_job_states: list[str] = Field(default_factory=list)
    wall_seconds: float | None = None
    shared_cluster: bool = False

    @computed_field
    @property
    def billed_seconds(self) -> float | None:
        """`billed_hours` converted to seconds, or `None` when there is no usage row.

        This, not `billed_envelope_ended - billed_envelope_started`, is what
        `billed_execution_gap_seconds` compares against `execution_seconds` — the
        envelope can overstate by an order of magnitude when a labelled cluster outlives
        the step that created it (see `usage.py`), and a gap computed against it would
        report a lingering cluster as the step's own waste rather than as the labelling
        defect it actually is.
        """
        if self.billed_hours is None:
            return None
        return self.billed_hours * 3600.0

    @computed_field
    @property
    def core_hours(self) -> float | None:
        """`core_seconds` in hours, or `None` when `core_seconds` is `None`."""
        if self.core_seconds is None:
            return None
        return self.core_seconds / 3600.0

    @computed_field
    @property
    def spot_share(self) -> float | None:
        """The spot fraction of `core_seconds`, or `None` when there is none to divide."""
        if not self.core_seconds:
            return None
        return self.spot_core_seconds / self.core_seconds if self.spot_core_seconds is not None else None

    @computed_field
    @property
    def billed_execution_gap_seconds(self) -> float | None:
        """`billed_seconds - execution_seconds`: cluster time paid for and not computing.

        `None` unless both `billed_seconds` and `execution_seconds` are present — never
        `0.0` as a stand-in for "not computable". See this module's docstring: this is the
        single most actionable number in the report.

        Also `None` when `shared_cluster` is True, regardless of whether both inputs are
        present: `billed_seconds` would then be the *instance's* whole billed time, not
        this step's own, and subtracting only this step's `execution_seconds` from it
        would overstate this step's waste by however much the instance's other steps
        computed — see this module's docstring and `cluster_compute_report`, which
        reports the instance's true idle time instead.
        """
        if self.shared_cluster or self.billed_seconds is None or self.execution_seconds is None:
            return None
        return self.billed_seconds - self.execution_seconds

    @computed_field
    @property
    def wall_execution_gap_seconds(self) -> float | None:
        """`wall_seconds - execution_seconds`: queueing under cluster contention.

        `None` unless both `wall_seconds` and `execution_seconds` are present.
        """
        if self.wall_seconds is None or self.execution_seconds is None:
            return None
        return self.wall_seconds - self.execution_seconds


@dataclass
class _UsageTotals:
    """One step's running totals across however many `StepUsage` rows named it.

    A plain, module-private accumulator -- not `StepCompute` itself -- because it is
    mutated in place while folding rows in, and `StepCompute` is meant to be the
    finished, immutable answer `compute_report` hands back.
    """

    net_cost: float | None = None
    currency: str | None = None
    billed_hour_buckets: set[datetime] = field(default_factory=set)
    billed_envelope_started: datetime | None = None
    billed_envelope_ended: datetime | None = None
    core_seconds: float | None = None
    spot_core_seconds: float | None = None
    machine_families: set[str] = field(default_factory=set)
    cluster_instances: set[str] = field(default_factory=set)

    @property
    def billed_hours(self) -> int | None:
        """Distinct hourly buckets billed, unioned across every row folded in.

        `None` exactly when no row has been folded in yet -- a step that billed at all
        always has at least one bucket, so an empty set here can only mean "no usage row
        seen", never "billed zero hours". See F7 in this project's review ledger: summing
        each row's own `billed_hours` instead would double-count an hour two rows (a
        `platform` row and a `ppp` row, say) both billed in.
        """
        return len(self.billed_hour_buckets) if self.billed_hour_buckets else None

    @property
    def billed_seconds(self) -> float | None:
        """`billed_hours` in seconds, or `None` under the same condition as `billed_hours`."""
        hours = self.billed_hours
        return None if hours is None else hours * 3600.0


def _accumulate_usage(totals: _UsageTotals, usage: StepUsage) -> None:
    """Fold one `StepUsage` row into a step's running totals, in place.

    Args:
        totals: The step's accumulator.
        usage: One usage row for that step.

    Raises:
        ValueError: If this usage row's currency disagrees with an earlier one for the
            same step. Summing across currencies would produce a number that looks like
            money and is not -- raising beats silently mixing them.
    """
    if totals.currency is not None and totals.currency != usage.currency:
        raise ValueError(
            f'step {usage.step!r} billed in more than one currency in the same report '
            f'({totals.currency!r} and {usage.currency!r}); compute_report cannot '
            'honestly sum a cost across currencies'
        )
    totals.currency = usage.currency
    totals.net_cost = usage.net_cost + (totals.net_cost or 0.0)
    totals.billed_hour_buckets.update(usage.billed_hour_buckets)
    totals.billed_envelope_started = (
        usage.started if totals.billed_envelope_started is None else min(totals.billed_envelope_started, usage.started)
    )
    totals.billed_envelope_ended = (
        usage.ended if totals.billed_envelope_ended is None else max(totals.billed_envelope_ended, usage.ended)
    )
    if usage.core_seconds is not None:
        totals.core_seconds = usage.core_seconds + (totals.core_seconds or 0.0)
    if usage.spot_core_seconds is not None:
        totals.spot_core_seconds = usage.spot_core_seconds + (totals.spot_core_seconds or 0.0)
    totals.machine_families.update(usage.machine_families)
    totals.cluster_instances.update(usage.cluster_instances)


def _step_totals(usages: Iterable[StepUsage]) -> dict[str, _UsageTotals]:
    """Fold every usage row into one `_UsageTotals` accumulator per step.

    Shared by `compute_report` and `cluster_compute_report`, so the two never derive a
    step's billed totals two different ways.
    """
    per_step_totals: dict[str, _UsageTotals] = {}
    for usage in usages:
        _accumulate_usage(per_step_totals.setdefault(usage.step, _UsageTotals()), usage)
    return per_step_totals


def _cluster_step_map(executions: Iterable[JobExecution]) -> dict[str, set[str]]:
    """Every step that submitted a Dataproc job to each cluster instance.

    From `dataproc.JobExecution` alone -- the one source that can see cluster reuse, since
    every job carries the *submitting* step's own label, unlike a cluster instance's
    billing rows, which repeat only the *creating* step's label for the instance's whole
    life (see this module's and `usage.py`'s docstrings). Every job state counts, not just
    `DONE`: a step with a job still `RUNNING` on an instance is really using it right now,
    which is exactly the contention `shared_cluster` exists to flag.
    """
    result: dict[str, set[str]] = {}
    for execution in executions:
        if execution.cluster_instance is None or execution.step is None:
            continue
        result.setdefault(execution.cluster_instance, set()).add(execution.step)
    return result


def _cluster_execution_totals(executions: Iterable[JobExecution]) -> dict[str, float]:
    """Summed `DONE` execution seconds per cluster instance, across every step that ran there.

    Mirrors `compute_report`'s own `_KEPT_JOB_STATE` rule at the instance level: a
    `CANCELLED`/`ERROR` job's CPU time is real spend but not kept work, so it is excluded
    here for the same reason it is excluded from a single step's `execution_seconds`.
    """
    totals: dict[str, float] = {}
    for execution in executions:
        cluster = execution.cluster_instance
        if cluster is None:
            continue
        if execution.state == _KEPT_JOB_STATE and execution.execution_seconds is not None:
            totals[cluster] = totals.get(cluster, 0.0) + execution.execution_seconds
    return totals


class ClusterCompute(BaseModel):
    """One Dataproc cluster instance's billed idle time, pooled across every step that used it.

    Exists because `StepCompute.billed_execution_gap_seconds` is `None` for a step flagged
    `shared_cluster` -- the step's own waste cannot be isolated from the instance's, but the
    instance's *total* idle time still can be, and this is where it is reported. See this
    module's docstring for the shape (`5b71ec48`, billed under `pts_reactome`, actually ran
    17 steps' jobs) this exists to make visible rather than lost.

    Args:
        cluster_instance: The `goog-dataproc-cluster-uuid` value.
        billing_step: The step whose billed usage this instance's `billed_seconds` is drawn
            from -- normally the step that created the instance, per `operators/dataproc.py`.
            `None` when no step's usage could be attributed to exactly this one instance
            (either nothing billed against it yet, or the billing step's own usage spans
            more than one instance and cannot be split between them -- see
            `usage.StepUsage.cluster_instances`).
        steps: Every step, sorted, that submitted at least one Dataproc job to this
            instance, from `dataproc.JobExecution` -- not from billing, which only ever
            names `billing_step`. Length 1 when the instance served only its own creator.
        currency: Currency `billed_seconds` is denominated in. `None` under exactly the
            condition `billing_step` is `None`.
        billed_seconds: `billing_step`'s billed time attributed to this instance. `None`
            under exactly the condition `billing_step` is `None`.
        execution_seconds: Summed `DONE` execution time across every job any step
            submitted to this instance. `None` if no job on this instance reached `DONE`.
    """

    cluster_instance: str
    billing_step: str | None = None
    steps: list[str] = Field(default_factory=list)
    currency: str | None = None
    billed_seconds: float | None = None
    execution_seconds: float | None = None

    @computed_field
    @property
    def idle_seconds(self) -> float | None:
        """`billed_seconds - execution_seconds`, or `None` unless both are present."""
        if self.billed_seconds is None or self.execution_seconds is None:
            return None
        return self.billed_seconds - self.execution_seconds

    @computed_field
    @property
    def shared(self) -> bool:
        """Whether more than one step submitted a Dataproc job to this instance."""
        return len(self.steps) > 1


def cluster_compute_report(
    usages: Iterable[StepUsage], executions: Iterable[JobExecution]
) -> list[ClusterCompute]:
    """Join billed cost and Dataproc execution time per cluster *instance*, not per step.

    Pure, like `compute_report`, and meant to be called alongside it on the same inputs --
    `compute_report` calls this internally to set `StepCompute.shared_cluster`, and a
    caller wanting the instance-level idle figure itself (`cli.py`'s `compute` command
    does) calls it directly.

    Args:
        usages: Billed usage rows, as `compute_report` takes.
        executions: Dataproc job records, as `compute_report` takes.

    Returns:
        One `ClusterCompute` per cluster instance named by either source, sorted by
        instance uuid. A GCE-only run -- no step's usage carries a `cluster_instance` and
        no execution carries one either -- returns `[]`.
    """
    per_step_totals = _step_totals(usages)
    cluster_steps = _cluster_step_map(executions)
    cluster_exec = _cluster_execution_totals(executions)

    cluster_billing: dict[str, tuple[str, float | None, str | None]] = {}
    for step, totals in per_step_totals.items():
        if len(totals.cluster_instances) != 1:
            continue
        (cluster,) = totals.cluster_instances
        cluster_billing[cluster] = (step, totals.billed_seconds, totals.currency)

    clusters = set(cluster_steps) | set(cluster_billing)
    return [
        ClusterCompute(
            cluster_instance=cluster,
            billing_step=cluster_billing[cluster][0] if cluster in cluster_billing else None,
            steps=sorted(cluster_steps.get(cluster, set())),
            currency=cluster_billing[cluster][2] if cluster in cluster_billing else None,
            billed_seconds=cluster_billing[cluster][1] if cluster in cluster_billing else None,
            execution_seconds=cluster_exec.get(cluster),
        )
        for cluster in sorted(clusters)
    ]


def compute_report(
    usages: Iterable[StepUsage],
    executions: Iterable[JobExecution],
    journal_events: Iterable[JournalEvent],
) -> list[StepCompute]:
    """Join billed cost, Dataproc execution time and journalled task wall time per step.

    Pure: does no I/O and holds no client. Callers fetch all three sources — typically
    `BillingExport.run_usage`, `dataproc.job_executions` and `Journal.read`, all scoped to
    the same run — and pass the results straight through.

    Args:
        usages: Billed usage rows, normally every row `BillingExport.run_usage` returned
            for one run. More than one row per step (distinct tool/product/currency) is
            summed rather than assumed impossible; see `_accumulate_usage` for the one
            case that is rejected outright rather than silently mixed.
        executions: Dataproc job records, normally every entry `dataproc.job_executions`
            returned for the same run. An execution whose `step` is `None` — a job with
            no `step` label whose id also matched no known step, see `dataproc.py` —
            cannot be joined to anything and is skipped here; `unresolved_job_count`
            on this same `executions` argument is how a caller surfaces that count
            instead of letting it vanish, since this function's own return value has
            nowhere to carry it.
        journal_events: The run's journal, normally `Journal.read()`'s result. Read once
            through `stall.baseline_from_journal` for `wall_seconds`; see this module's
            docstring for what that reuse means for a step re-run within the same run.

    Returns:
        One `StepCompute` per step named by any of the three sources, sorted by step
        name — the only ordering key every row is guaranteed to carry, since a step can
        be missing from any two of the three sources.
    """
    usages = list(usages)
    executions = list(executions)
    wall_by_step = baseline_from_journal(list(journal_events))

    per_step_totals = _step_totals(usages)

    job_states: dict[str, list[str]] = {}
    execution_totals: dict[str, float] = {}
    for execution in executions:
        if execution.step is None:
            continue
        per_step_totals.setdefault(execution.step, _UsageTotals())
        job_states.setdefault(execution.step, []).append(execution.state)
        if execution.state == _KEPT_JOB_STATE and execution.execution_seconds is not None:
            execution_totals[execution.step] = execution_totals.get(execution.step, 0.0) + execution.execution_seconds

    for step in wall_by_step:
        per_step_totals.setdefault(step, _UsageTotals())

    shared_steps = {
        cluster.billing_step
        for cluster in cluster_compute_report(usages, executions)
        if cluster.shared and cluster.billing_step is not None
    }
    # A step whose usage spans more than one cluster instance is just as unable to support
    # a per-step waste figure as a step on a shared one -- `billed_seconds` would mix two
    # physical instances' billed time into one number, which is not a duration on either of
    # them. Folded into the same flag rather than a second one: both cases share the one
    # consequence a reader needs to know -- this step's own `billed_execution_gap_seconds`
    # cannot be trusted -- and `cluster_compute_report` already declines to guess a
    # `billed_seconds` for either kind of ambiguity (see `ClusterCompute.billing_step`).
    ambiguous_steps = {step for step, totals in per_step_totals.items() if len(totals.cluster_instances) > 1}
    shared_steps |= ambiguous_steps

    return [
        StepCompute(
            step=step,
            net_cost=totals.net_cost,
            currency=totals.currency,
            billed_hours=totals.billed_hours,
            billed_envelope_started=totals.billed_envelope_started,
            billed_envelope_ended=totals.billed_envelope_ended,
            core_seconds=totals.core_seconds,
            spot_core_seconds=totals.spot_core_seconds,
            machine_families=sorted(totals.machine_families),
            execution_seconds=execution_totals.get(step),
            dataproc_job_states=sorted(job_states.get(step, [])),
            wall_seconds=wall_by_step.get(step),
            shared_cluster=step in shared_steps,
        )
        for step, totals in sorted(per_step_totals.items())
    ]
