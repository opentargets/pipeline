"""Deciding whether a running task has stalled.

Detection degrades rather than failing closed. Most steps have no observed history
on early runs — the billing export holds at most 18 of roughly 150 steps, and
Airflow's own history is destroyed with the VM — so a step with history is judged
against its own observed maximum and a step without one against a single absolute
ceiling. The verdict records which rule fired, because "four times its usual" and
"past the blanket limit" are different things to tell a human.

Elapsed time here is task wall time, which includes queueing. That is deliberate: a
task stuck for hours needs attention whether it is hung or waiting for a busy
cluster, and separating the two is the compute report's job, not the alarm's.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel

from orchestration.supervisor.airflow import TaskInstance
from orchestration.supervisor.journal import JournalEvent
from orchestration.utils.common import STALL_CEILING_SECONDS, STALL_MULTIPLIER

Baseline = dict[str, float]
"""Observed maximum duration in seconds, per step."""

_ACTIVE_STATES = frozenset({'running', 'deferred', 'restarting'})
"""States in which a task is holding resources and can therefore stall.

`deferred` matters as much as `running` here: every PIS and PTS step runs through
`ComputeEngineRunContainerizedWorkloadSensor` with `deferrable=True`
(`dags/unified_pipeline.py:110,190`), so a hung GCE VM parks the task in `deferred`
and never reaches `running` again. Judging only `running` would leave the pipeline's
most common step type invisible to stall detection.

`restarting` matters for the same reason. It is not terminal — it is Airflow's state
for a task that *was* running and got externally interrupted, cleared while running,
or lost its worker. The task instance is still alive, awaiting reconciliation, with a
`start_date` from before the interruption, and it still holds whatever cloud resource
it had when interrupted: no operator in this codebase implements `on_kill`
(`grep -rn on_kill orchestration/src/orchestration/operators/` returns nothing), so
nothing reclaims that resource on its own. Unlike `up_for_retry` there is no bound on
how long a task can sit `restarting`.

`up_for_retry` stays out on purpose: it is reachable only for the `stage_jar_*` tasks,
bounded to roughly six minutes across their three retries, so flagging it would be
noise. `queued` and `scheduled` stay out because they are absent from this set — those
task instances have no `start_date` yet either, so they would also be skipped by the
`start_date is None` guard below, but that guard is not why they are excluded here;
whether long queueing deserves its own alarm is a question for a later phase, not
this one.
"""


class StallVerdict(BaseModel):
    """A running task judged to have stalled.

    Args:
        task_id: The task instance's id.
        elapsed: Seconds since it started, including any queueing.
        threshold: The threshold it passed.
        basis: Which rule fired — `history` for a step with observed runs,
            `ceiling` for one without.
    """

    task_id: str
    elapsed: float
    threshold: float
    basis: Literal['history', 'ceiling']


def baseline_from_journal(events: list[JournalEvent]) -> Baseline:
    """Build a per-step baseline from journalled completions.

    Args:
        events: The run journal, or several runs' journals concatenated.

    Returns:
        The observed maximum duration per step. Steps never seen are absent, which is
        what makes the ceiling fallback necessary rather than optional. A completion
        whose `duration` cannot be read as a number is skipped rather than raising, so
        one malformed event degrades to a partial baseline instead of losing every
        step's history.
    """
    baseline: Baseline = {}
    for event in events:
        if event.event_type != 'step_completed' or event.step is None:
            continue
        duration = event.payload.get('duration')
        if duration is None:
            continue
        try:
            seconds = float(duration)
        except (TypeError, ValueError):
            continue
        baseline[event.step] = max(baseline.get(event.step, 0.0), seconds)
    return baseline


def stalled(
    task: TaskInstance,
    baseline: Baseline,
    now: datetime,
    ceiling: float = STALL_CEILING_SECONDS,
    multiplier: float = STALL_MULTIPLIER,
) -> StallVerdict | None:
    """Judge whether a running task has stalled.

    Args:
        task: The task instance to judge.
        baseline: Observed maxima per step.
        now: The current time, injected so the decision is testable.
        ceiling: Absolute threshold for a step with no history.
        multiplier: How far past its observed maximum a known step may run.

    Returns:
        A verdict, or None if the task is not in an active state (`running`, `deferred`,
        `restarting`), has no start date, or is within its threshold.
    """
    if task.state not in _ACTIVE_STATES or task.start_date is None:
        return None

    elapsed = (now - task.start_date).total_seconds()
    observed = baseline.get(task.task_id)
    threshold = observed * multiplier if observed is not None else ceiling
    basis: Literal['history', 'ceiling'] = 'history' if observed is not None else 'ceiling'

    if elapsed <= threshold:
        return None
    return StallVerdict(task_id=task.task_id, elapsed=elapsed, threshold=threshold, basis=basis)
