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
        billed_hours: Summed `StepUsage.billed_hours` — distinct hourly buckets billed,
            not an envelope. `None` under the same condition as `net_cost`.
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
        """
        if self.billed_seconds is None or self.execution_seconds is None:
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
    billed_hours: int | None = None
    billed_envelope_started: datetime | None = None
    billed_envelope_ended: datetime | None = None
    core_seconds: float | None = None
    spot_core_seconds: float | None = None
    machine_families: set[str] = field(default_factory=set)


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
    totals.billed_hours = usage.billed_hours + (totals.billed_hours or 0)
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
            returned for the same run. An execution whose `step` is `None` — a job id
            that matched no known step — cannot be joined to anything and is skipped;
            `dataproc.job_executions` is what reports it, this function is not the place
            that would silently drop it.
        journal_events: The run's journal, normally `Journal.read()`'s result. Read once
            through `stall.baseline_from_journal` for `wall_seconds`; see this module's
            docstring for what that reuse means for a step re-run within the same run.

    Returns:
        One `StepCompute` per step named by any of the three sources, sorted by step
        name — the only ordering key every row is guaranteed to carry, since a step can
        be missing from any two of the three sources.
    """
    wall_by_step = baseline_from_journal(list(journal_events))

    per_step_totals: dict[str, _UsageTotals] = {}
    for usage in usages:
        _accumulate_usage(per_step_totals.setdefault(usage.step, _UsageTotals()), usage)

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
        )
        for step, totals in sorted(per_step_totals.items())
    ]
