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

**A job id encodes its step, but not at a fixed position.** `up-pts-f5014-
pts_drug_molecule-c516h` and `up-pts-literature-f5014-pts_literature_ontoma-5znfv`
both carry a real step name, but the cluster-name prefix before it varies in length
(`SubmitJobOperator.execute` in `operators/dataproc.py` builds the id as
`f'{cluster_name}-{step_name}-{random_id()}'`, and `cluster_name` is not fixed
width). Splitting on `-` and indexing works for one shape and silently mis-slices the
other. `step_for_job_id` instead matches the id against
`datasets.unified_pipeline_steps()` and keeps the longest match, which also resolves
the one real ambiguity this creates: a step name that is itself a substring of
another step name (`pts_disease` inside `pts_disease_hpo`).

**A step can have several jobs in one run.** `pts_drug_molecule` in the run verified
above has exactly this: a `CANCELLED` job (`...-gttbk`) beside the `DONE` job that
succeeded (`...-c516h`) -- an observed re-run, not a theoretical case. `job_executions`
reports every job Dataproc returns, one `JobExecution` each, and does not sum, dedupe,
or pick a "winning" job at this layer. Collapsing here would silently discard
whichever jobs lost, and there is no one correct choice between summing, taking the
successful attempt, or keeping both without knowing what the number is for. That
choice belongs to `compute.py`'s join, which has the full list -- and the other two
sources -- in hand to make it.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol

from pydantic import BaseModel

from orchestration.supervisor.datasets import unified_pipeline_steps

if TYPE_CHECKING:
    from collections.abc import Iterable

_TERMINAL_STATES = frozenset({'DONE', 'ERROR', 'CANCELLED'})
"""`JobStatus.State` values that mean a job has stopped running for good.

`CANCEL_PENDING` and `CANCEL_STARTED` are deliberately excluded: the job is still
transitioning and has not yet reached the state `status.state_start_time` would need
to mean "when it stopped" for `job_execution`'s subtraction to be meaningful.
"""


class JobExecution(BaseModel):
    """One Dataproc job's actual compute time, as distinct from how long it was billed.

    Args:
        job_id: The Dataproc job id, e.g. `up-pts-f5014-pts_drug_molecule-c516h`.
        step: The `unified_pipeline.yaml` step name recovered from `job_id`, or None
            if no known step name appears in it. Reported rather than dropped: a
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
    """

    job_id: str
    step: str | None
    state: str
    execution_seconds: float | None = None
    started: datetime | None = None
    ended: datetime | None = None


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


class JobLike(Protocol):
    """The part of a Dataproc `Job` this module reads."""

    @property
    def reference(self) -> _ReferenceLike: ...

    @property
    def status(self) -> _StatusLike: ...

    @property
    def status_history(self) -> Iterable[_StatusLike]: ...


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
    """Recover a `unified_pipeline.yaml` step name from a job id.

    Matches by substring rather than by position, since the cluster-name prefix
    before the step name is not fixed width -- see this module's docstring. When more
    than one known step name is a substring (a shorter step name that is itself a
    prefix of a longer one, e.g. `pts_disease` inside `pts_disease_hpo`), the longest
    match wins, since it is the more specific -- and, being longer, correct -- answer.

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

    Args:
        job: The job, as returned by the Job Controller API, or a stand-in with the
            same shape.
        known_steps: Step names to match the job id against, passed to
            `step_for_job_id`. Defaults to `datasets.unified_pipeline_steps()`.

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
    return JobExecution(
        job_id=job_id,
        step=step_for_job_id(job_id, steps),
        state=state,
        execution_seconds=execution_seconds,
        started=started,
        ended=ended,
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
