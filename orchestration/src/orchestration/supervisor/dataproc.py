"""Execution time for a Dataproc job, read from the Job Controller API.

Wall time (Airflow) and billed span (the billing export) both bound a step from the
outside. Neither says how much of that time was actual compute: `usage.py`'s own
docstring is explicit that hour-bucketed billing rows cannot support a duration.
This module reads the one measurement that can -- the interval Dataproc's own job
history records between a job entering `RUNNING` and reaching whatever state it
reached next.

**The terminal state lives on `status`, not in `status_history`.** Verified live,
2026-08-24, against `up-pts-f5014-pts_drug_molecule-c516h` in region `europe-west1`,
project `open-targets-eu-dev`: `status_history` carried `PENDING`, `SETUP_DONE`,
`RUNNING`, each with its own `state_start_time` -- but the job's final state (`DONE`
in that case) was never appended to that list. It was only on `status` itself.
Execution time is therefore `status.state_start_time` minus the `RUNNING` entry's
`state_start_time` from `status_history`, not a difference taken entirely from one
place.

**The job's own `step` label is authoritative; matching the id against known steps is
only a fallback.** Every job `SubmitJobOperator` submits carries a `step` label (see
`operators/dataproc.py`'s `execute` and `utils/labels.py`), set once at submission
time from the same `step_name` that names the job in `unified_pipeline.yaml` -- so
`job.labels['step']` is ground truth, not a guess. `step_for_job_id`, matching the id
against `known_steps`, exists only for a job with no `step` label, which by
construction should not happen for anything `SubmitJobOperator` submits, but is kept
as a fallback rather than an assumption. This distinction was not academic: on a
verified run (`up-20260527-1458`), `pts_target_safety` had since grown a same-run
sibling step, `pts_target`, whose name is a prefix of the older job ids' cluster
segment; two real jobs matched `pts_target` by substring while their own labels said
`pts_target_safety`, and two `etl_literature` jobs matched no `known_steps` entry at
all (the yaml had renamed away from it) and were dropped entirely by the
substring-only path. Reading the label first gets both right regardless of what the
*current* checkout's yaml happens to contain, which id-matching alone can never
guarantee for a run from a different point in time.

**A job id encodes its step, but not at a fixed position** -- this is what the
id-matching fallback has to work around. `up-pts-f5014-pts_drug_molecule-c516h` and
`up-pts-literature-f5014-pts_literature_ontoma-5znfv` both carry a real step name, but
the cluster-name prefix before it varies in length (`SubmitJobOperator.execute` in
`operators/dataproc.py` builds the id as `f'{cluster_name}-{step_name}-
{random_id()}'`, and `cluster_name` is not fixed width). Splitting on `-` and
indexing works for one shape and silently mis-slices the other. `step_for_job_id`
instead matches the id against `datasets.unified_pipeline_steps()` and keeps the
longest match, which also resolves the one real ambiguity this creates: a step name
that is itself a substring of another step name (`pts_disease` inside
`pts_disease_hpo`) -- a resolution that only holds when both names are still in the
yaml, which is exactly why it is a fallback and not the primary path.

**A step can have several jobs in one run.** `pts_drug_molecule` in the run verified
above has exactly this: a `CANCELLED` job (`...-gttbk`) beside the `DONE` job that
succeeded (`...-c516h`) -- an observed re-run, not a theoretical case. `job_executions`
reports every job Dataproc returns, one `JobExecution` each, and does not sum, dedupe,
or pick a "winning" job at this layer. Collapsing here would silently discard
whichever jobs lost, and there is no one correct choice between summing, taking the
successful attempt, or keeping both without knowing what the number is for. That
choice belongs to `compute.py`'s join, which has the full list -- and the other two
sources -- in hand to make it.

**`job.placement.cluster_uuid` is the same instance `usage.py` bills under
`goog-dataproc-cluster-uuid`.** `operators/dataproc.py` submits jobs with
`use_if_exists=True`, so one cluster instance can carry jobs from several steps while
its billing rows -- set once, at creation -- still name only the step that created it
(see `usage.py`'s module docstring on `shared_cluster`, and F1 in this project's
review ledger: the guard billing alone can compute is structurally unable to see
this). Reading `job.placement.cluster_uuid` here, alongside the job's own `step`
label, is what lets `compute.py` reconstruct, from Dataproc's own record, every step
that actually ran on a given instance -- the one join billing cannot do by itself.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol

from pydantic import BaseModel

from orchestration.supervisor.datasets import unified_pipeline_steps

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

_TERMINAL_STATES = frozenset({'DONE', 'ERROR', 'CANCELLED'})
"""`JobStatus.State` values that mean a job has stopped running for good.

`CANCEL_PENDING` and `CANCEL_STARTED` are deliberately excluded: the job is still
transitioning and has not yet reached the state `status.state_start_time` would need
to mean "when it stopped" for `job_execution`'s subtraction to be meaningful.
"""

TERMINAL_JOB_STATES = _TERMINAL_STATES
"""Public alias of `_TERMINAL_STATES`, for callers outside this module that need to
tell a job still in progress apart from one that has stopped for good -- e.g.
`cli.py`'s `compute` command footer, which must not call a `RUNNING`, `PENDING` or
`SETUP_DONE` job "cancelled or errored"."""


class JobExecution(BaseModel):
    """One Dataproc job's actual compute time, as distinct from how long it was billed.

    Args:
        job_id: The Dataproc job id, e.g. `up-pts-f5014-pts_drug_molecule-c516h`.
        step: The `unified_pipeline.yaml` step name, read from the job's own `step`
            label when present (see `job_execution`) and otherwise recovered from
            `job_id` by substring match, or None if the label is absent and no known
            step name appears in the id either. Reported rather than dropped: a
            silently-skipped job would understate its step's execution time with
            nothing to show anything was missing.
        state: The job's current or terminal state name (`DONE`, `ERROR`,
            `CANCELLED`, `RUNNING`, ...), read off `JobStatus.State`.
        execution_seconds: Time actually spent computing, or None when it cannot be
            derived -- the job never reached `RUNNING` (see `started`), or it has not
            yet reached a terminal state. None, never 0.0: a job that has not
            finished executing has not "executed in zero seconds".
        started: When the job entered `RUNNING`, from `status_history`, or None if it
            never did (e.g. it failed during setup).
        ended: When the job reached its terminal state, from
            `status.state_start_time`, or None while the job is still in progress.
        cluster_instance: `job.placement.cluster_uuid` -- the same value `usage.py`
            reads off `goog-dataproc-cluster-uuid`. None only if the API ever omits
            it, which has not been observed; every submitted job is placed on some
            cluster. This, not the job id or the cluster name, is what lets
            `compute.py` tell "this instance served only this step" apart from "this
            instance also carried other steps' jobs" -- see this module's docstring.
    """

    job_id: str
    step: str | None
    state: str
    execution_seconds: float | None = None
    started: datetime | None = None
    ended: datetime | None = None
    cluster_instance: str | None = None


class _StatusLike(Protocol):
    """The part of a Dataproc `JobStatus` this module reads.

    Declared as properties rather than plain attributes: a plain-attribute protocol
    member is matched invariantly, which would reject any concrete class other than
    one whose attribute types are exactly `_StatusLike`/`_ReferenceLike` themselves --
    defeating structural typing for both the real proto-plus `Job` and every test
    double. A read-only property is matched covariantly instead, which is what
    structural duck typing here actually needs.
    """

    @property
    def state(self) -> Any: ...

    @property
    def state_start_time(self) -> datetime | None: ...


class _ReferenceLike(Protocol):
    """The part of a Dataproc `JobReference` this module reads."""

    @property
    def job_id(self) -> str: ...


class _PlacementLike(Protocol):
    """The part of a Dataproc `JobPlacement` this module reads."""

    @property
    def cluster_uuid(self) -> str: ...


class JobLike(Protocol):
    """The part of a Dataproc `Job` this module reads."""

    @property
    def reference(self) -> _ReferenceLike: ...

    @property
    def status(self) -> _StatusLike: ...

    @property
    def status_history(self) -> Iterable[_StatusLike]: ...

    @property
    def labels(self) -> Mapping[str, str]: ...

    @property
    def placement(self) -> _PlacementLike: ...


class Client(Protocol):
    """The part of a Dataproc `JobControllerClient` this module uses."""

    def list_jobs(self, *, project_id: str, region: str, filter: str) -> Iterable[JobLike]:
        """List jobs matching a filter, as `JobControllerClient.list_jobs` does."""
        ...


def _state_name(state: Any) -> str:
    """The state's string name.

    Works for both the real `JobStatus.State` enum, whose `str()` gives its integer
    value rather than its name, and a plain string test double, which has no `.name`
    to read.

    Args:
        state: A `JobStatus.State` member, or a plain string standing in for one.

    Returns:
        The state's name, e.g. `'DONE'`.
    """
    name = getattr(state, 'name', None)
    return name if name is not None else str(state)


def step_for_job_id(job_id: str, known_steps: Iterable[str]) -> str | None:
    """Recover a `unified_pipeline.yaml` step name from a job id, by substring match.

    This is the fallback path used only when a job carries no `step` label -- see
    `job_execution` and this module's docstring for why the label, not this function,
    is the primary source. Matches by substring rather than by position, since the
    cluster-name prefix before the step name is not fixed width. When more than one
    known step name is a substring (a shorter step name that is itself a prefix of a
    longer one, e.g. `pts_disease` inside `pts_disease_hpo`), the longest match wins,
    since it is the more specific -- and, being longer, correct -- answer. This
    resolution only holds while `known_steps` still contains both names; a step
    renamed out of the current yaml can no longer be told apart from a shorter
    surviving prefix, which is exactly the failure the `step` label sidesteps.

    Args:
        job_id: The Dataproc job id.
        known_steps: Step names to match against, e.g. `datasets.unified_pipeline_steps()`.

    Returns:
        The longest known step name that appears in `job_id`, or None if none does.
    """
    matches = [step for step in known_steps if step in job_id]
    return max(matches, key=len) if matches else None


def job_execution(job: JobLike, known_steps: Iterable[str] | None = None) -> JobExecution:
    """Convert one Dataproc job into its execution record.

    Reads the step from `job.labels['step']` first -- the value `SubmitJobOperator`
    itself set at submission time, so it names the step that job actually ran for,
    regardless of what the current checkout's `unified_pipeline.yaml` says. Only when
    that label is absent or empty does this fall back to `step_for_job_id`, matching
    the id against `known_steps` -- see this module's and `step_for_job_id`'s
    docstrings for why that fallback can silently disagree with the truth for an old
    or renamed run.

    Args:
        job: The job, as returned by the Job Controller API, or a stand-in with the
            same shape.
        known_steps: Step names to match the job id against when `job` carries no
            `step` label, passed to `step_for_job_id`. Defaults to
            `datasets.unified_pipeline_steps()`.

    Returns:
        The job's execution record.
    """
    steps = list(known_steps) if known_steps is not None else unified_pipeline_steps()
    job_id = job.reference.job_id
    state = _state_name(job.status.state)
    started = next(
        (entry.state_start_time for entry in job.status_history if _state_name(entry.state) == 'RUNNING'),
        None,
    )
    ended = job.status.state_start_time if state in _TERMINAL_STATES else None
    execution_seconds = (ended - started).total_seconds() if started is not None and ended is not None else None
    step = job.labels.get('step') or None
    if step is None:
        step = step_for_job_id(job_id, steps)
    return JobExecution(
        job_id=job_id,
        step=step,
        state=state,
        execution_seconds=execution_seconds,
        started=started,
        ended=ended,
        cluster_instance=job.placement.cluster_uuid or None,
    )


def job_executions(
    client: Client,
    project: str,
    region: str,
    run: str,
    known_steps: Iterable[str] | None = None,
) -> list[JobExecution]:
    """List every Dataproc job billed to one run and convert each to its execution record.

    Filters server-side on the `run` label, the same cleaned value
    `usage.StepUsage.run` carries (both go through `Labels.add_dag_run_id`, which
    cleans through the same `clean_label`) -- so a caller with a `StepUsage.run`
    value in hand can pass it straight through unchanged.

    Args:
        client: An authenticated Job Controller client, or an injected stand-in.
            Injected so tests never need credentials, matching `journal.py` and
            `gcs.py`.
        project: The GCP project the jobs were submitted in.
        region: The Dataproc region.
        run: The `run` label value to filter on.
        known_steps: Passed through to `job_execution` for every job. Read once here
            via `datasets.unified_pipeline_steps()` when omitted, rather than once
            per job, so a run with many jobs does not re-parse the yaml repeatedly.

    Returns:
        One `JobExecution` per job Dataproc returns, unaggregated. See this module's
        docstring for why a step with several jobs yields several entries here rather
        than one.
    """
    steps = list(known_steps) if known_steps is not None else unified_pipeline_steps()
    jobs = client.list_jobs(project_id=project, region=region, filter=f'labels.run = {run}')
    return [job_execution(job, steps) for job in jobs]
