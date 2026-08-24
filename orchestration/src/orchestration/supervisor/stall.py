"""Deciding whether a running task has stalled.

Detection degrades rather than failing closed. A step is judged against its own
observed maximum when one is available, and against a single absolute ceiling when it
is not. The ceiling is the rule in practice, not the fallback: `baseline_from_journal`
can only build an observed maximum from a `step_completed` event earlier in the *same
run's* journal, and a step that has completed leaves `_ACTIVE_STATES` and is never
judged again — so history only ever fires for a step cleared and re-run within one run
(see `baseline_from_journal` for that case, and for why a baseline spanning prior runs
is deferred rather than built now). The verdict records which rule fired, because "two
and a half times its usual" and "past the blanket limit" are different things to tell a
human.

Most steps never accumulate the observed history to benefit from the history rule: the
billing export holds at most 18 of the pipeline's 132 steps, and Airflow's own history
is destroyed with the VM.

The baseline is only ever consulted for a step's own execution task, never for a
sibling in its group. `step_from_task_id` collapses every task in a group onto the
same step name (`pts_target.delete_vm_pts_target` and `pts_target.run_pts_target`
both map to `pts_target`), so an ungated lookup would hand the run task's journalled
duration to every sibling — a `diff_` task doing a slow GCS listing judged stalled on
a fabricated history basis, or a `delete_vm_` task whose step has a 3-hour observed
maximum getting a 7.5-hour threshold (2.5x that maximum) instead of the 6-hour
ceiling, delaying a real hang report by 1.5 hours. `stalled` gates the lookup with
`is_run_task` for exactly this reason.

Elapsed time here is measured from `start_date` — when the task began executing —
and therefore excludes queueing. That is deliberate, not an oversight: the baseline
this elapsed time is compared against comes from Airflow's own `duration`, which is
also execution time only, not wall time from `queued_dttm`. Measuring elapsed from
`queued_dttm` instead would look like the more thorough fix and would in fact be a
bug — it would compare a queue-inclusive elapsed against a queue-exclusive baseline,
so any step that reliably queues for a while would eventually cross its threshold on
queueing alone and false-alarm forever. Detecting a task stuck queueing rather than
stuck executing is a real and separate failure mode — `queued_dttm` is available on
`TaskInstance` for it — but it needs its own rule and its own baseline, deferred to a
later phase rather than folded in here where it would corrupt this one.

`stalled` is judged per task, and only once a task has actually started: a task frozen
in `running` still has a `start_date`, its elapsed time keeps growing, and `stalled`
still flags it once it crosses its threshold — a scheduler that dies mid-task does not
leave that task invisible. The gap is earlier than that: a task instance stuck in
`queued` or `scheduled` has no `start_date` yet, so it never enters `_ACTIVE_STATES`
and `stalled` has nothing to judge it against for as long as it sits there. If the
scheduler dies before handing a task off to `running` — or between two steps, with
nothing yet queued for the next one — that task, and everything queued behind it,
simply never gets a verdict, while Dataproc clusters already provisioned for earlier
steps keep billing. `run_stalled` is the run-level complement, covering two
signatures `stalled` cannot see: a run that is technically active but has produced no
step-level news in a long while (`'no_progress'`), and a run with nothing active at all
while work remains pending behind it (`'stuck_trigger'`) — see its docstring for both.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel

from orchestration.supervisor.airflow import TaskInstance
from orchestration.supervisor.journal import JournalEvent, is_heartbeat
from orchestration.supervisor.step_identity import is_run_task, step_from_task_id
from orchestration.utils.common import STALL_CEILING_SECONDS, STALL_MULTIPLIER

Baseline = dict[str, float]
"""Observed maximum duration in seconds, keyed by the bare `unified_pipeline.yaml` step name.

See `baseline_from_journal` for why the key is the bare step name and not the
fully-qualified `task_id`.
"""

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
        task_id: The task instance's ref — `task_id`, qualified with `map_index` for a
            mapped instance (see `TaskInstance.ref`) — so two shards of the same mapped
            task never produce indistinguishable verdicts.
        elapsed: Seconds since `start_date`, i.e. execution time only, excluding
            queueing — see the module docstring for why.
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

    `JournalEvent.step` holds the bare `unified_pipeline.yaml` step name (`pts_target`),
    the same spelling GCP uses as its billing label and `usage.StepUsage.step` already
    carries. That is deliberately not the fully-qualified Airflow `task_id`
    (`pts_target.run_pts_target`): every step in this pipeline runs inside a
    `@task_group(group_id=step_name)` with `prefix_group_id` defaulting to True, so the
    two spellings differ for every step, not just some. `stalled` converts the task id
    it is handed down to the bare step name (`step_from_task_id`) before looking up the
    baseline — but only for a step's own run task, since `is_run_task` gates both the
    conversion and the lookup (see the module docstring) — so both sides of the lookup
    agree whenever the lookup happens at all.

    The direction matters because a mismatch here fails silently: a baseline keyed in a
    namespace `stalled` never looks up would miss on every lookup, falling back to the
    ceiling — not raise, not warn. That symptom alone, `basis == 'ceiling'` on every
    verdict, discriminates nothing: it is also today's ordinary, correct behaviour,
    since the history rule only ever fires for a step cleared and re-run within the same
    run (see the module docstring) and most steps are never re-run. That is why
    `tests/test_supervisor_snapshot.py` pins both directions of the key directly, rather
    than relying on this symptom to surface a mismatch in production.

    Args:
        events: One run's journal, as returned by `Journal.read()`. `step` on each
            `step_completed` event must be the bare step name, as above. A baseline
            spanning several runs would make the history rule the common case instead
            of the narrow one, but building it needs to enumerate prior runs' journal
            prefixes — keyed on `dag_run_id`, per `journal.py`'s module docstring.
            Nothing in this codebase enumerates prior runs' journals yet; deferred
            until that need arises.

    Returns:
        The observed maximum duration per step (keyed by the bare step name).
        Steps never seen are absent, which is what makes the ceiling fallback necessary
        rather than optional. A completion whose `duration` cannot be read as a number
        is skipped rather than raising, so one malformed event degrades to a partial
        baseline instead of losing every step's history.
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

    Only the step's own execution task is judged against its history; a sibling task in
    the same group always falls to the ceiling, since the baseline holds durations for
    the execution task alone and `step_from_task_id` would otherwise hand its duration
    to every sibling — see the module docstring.

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
    observed = baseline.get(step_from_task_id(task.task_id)) if is_run_task(task.task_id) else None
    threshold = observed * multiplier if observed is not None else ceiling
    basis: Literal['history', 'ceiling'] = 'history' if observed is not None else 'ceiling'

    if elapsed <= threshold:
        return None
    return StallVerdict(task_id=task.ref, elapsed=elapsed, threshold=threshold, basis=basis)


_PENDING_STATE = 'pending'
"""What `snapshot.take_snapshot` counts a task instance with no Airflow state as.

Duplicated from `snapshot._PENDING` rather than imported: `snapshot.py` imports from
this module (`stalled`, `baseline_from_journal`), so importing back from `snapshot.py`
here would be a cycle. The value is a plain, stable literal — Airflow never reports it
itself, `snapshot.py` invents it — so duplicating the one string is a smaller risk than
the import cycle would be."""

_STUCK_TRIGGER_MOVING_STATES = frozenset({'queued', 'scheduled', 'up_for_retry'})
"""States that mean a task instance is on its way toward `running`, not stuck.

`'stuck_trigger'`'s own docstring describes the shape it looks for as "nothing
executing, nothing queued to execute it, work still waiting" — but until this constant
was introduced, `run_stalled` only ever checked the first and third clauses: `active`
counted `_ACTIVE_STATES` alone, and anything not `running`/`deferred`/`restarting` fell
straight through to `_PENDING_STATE`, indistinguishable from a task Airflow has never
touched. That produced false positives on every one of a run's ordinary shapes:

- A run's opening minutes: the scheduler creates all ~132 task instances as `pending`
  and moves the roots through `scheduled` -> `queued` before anything first reaches
  `running`.
- A routine step hand-off, any time later in the run: the next step's task sits
  `scheduled` for a moment between one step finishing and the next starting.
- `up_for_retry`, reachable only for `stage_jar_*` (bounded to roughly six minutes
  across its three retries at a 2-minute delay, `unified_pipeline.py:299` — `max_tries`
  is 0 everywhere else in this pipeline) — a scheduled retry, not an abandoned task.

All three mean the scheduler is actively moving this task toward execution; none of
them is the "trigger rule that never fired" `'stuck_trigger'` exists to catch. A task
sitting in one of these states does not, on its own, prove the scheduler is dead —
unlike a task that is neither active, moving, nor even touched at all
(`_PENDING_STATE`), which has nothing else that could still be about to move it."""

_RUN_STALL_WAKEUP_THRESHOLD = 6
"""Consecutive silent wakeups before `'no_progress'` fires — one hour, at the 10-minute
cadence `deployment/startup_machine.sh`'s `CRON_LINE` actually runs the observer on.

Counted in wakeups rather than minutes so a cron that was itself down for a while does
not, on its own, manufacture a false alarm the moment it comes back — see
`_wakeups_since_step_event`.

An hour is long enough that a step legitimately in progress, with nothing else running
alongside it, does not immediately misfire on it: `run_stalled` additionally never
fires while any active task has not itself crossed its own `stalled()` threshold,
which is what makes a merely slow-but-healthy task safe against this constant
regardless of its value. It does not, and structurally cannot, fire *before* any
individual task reaches its own ceiling: the `len(stalls) >= active` gate means every
currently active task must already have been individually flagged by `stalled()` —
which for a step with no history means it has already passed `STALL_CEILING_SECONDS`'s
6h — before `'no_progress'` is even eligible to fire. The two are sequential, not a
race: `stalled()` reports first, `'no_progress'` can only ever confirm afterwards that
the *whole* active set, not just one flaky task, is stuck. That
gate — not this number — is what actually protects a legitimately long step; this
number only decides how long the run stays quiet once every active task has *already*
been individually flagged."""


class RunStallVerdict(BaseModel):
    """The run as a whole judged to have stalled — a different thing from any one task.

    Args:
        reason: Which signature fired. `'no_progress'`: the run is `running`, but no
            `step_completed`/`step_failed` event has been journalled for
            `_RUN_STALL_WAKEUP_THRESHOLD` wakeups, and every currently active task is
            already individually flagged by `stalled()` — a scheduler that has stopped
            advancing the run, or a hung task past the point `stalled()` itself would
            already report. `'stuck_trigger'`: the run is `running`, no task is active
            (`running`, `deferred`, `restarting`) or moving toward active
            (`_STUCK_TRIGGER_MOVING_STATES`: `queued`, `scheduled`, `up_for_retry`), yet
            tasks remain pending — a trigger rule that never fired, or a scheduler that
            has given up entirely. See `run_stalled` for the full reasoning behind both.
        wakeups: For `'no_progress'`, how many consecutive wakeups have journalled no
            step event, from `_wakeups_since_step_event`. `None` for `'stuck_trigger'`,
            which is judged from a single snapshot and needs no wakeup history.
        active_tasks: For `'no_progress'`, how many tasks are currently active — all of
            them already carrying their own `stalled()` verdict (see `reason`). `None`
            for `'stuck_trigger'`, which by definition has zero active tasks.
        pending: For `'stuck_trigger'`, how many tasks are waiting to run. `None` for
            `'no_progress'`, which does not care whether anything is pending.
    """

    reason: Literal['no_progress', 'stuck_trigger']
    wakeups: int | None = None
    active_tasks: int | None = None
    pending: int | None = None


def _wakeups_since_step_event(events: list[JournalEvent]) -> int:
    """Count consecutive heartbeats, working backwards, since the last step-level event.

    Walks `events` from the most recent entry backwards — `Journal.read()` returns them
    sorted chronologically ascending by `at` — counting heartbeats (`journal.is_heartbeat`)
    until a `step_completed` or `step_failed` event is reached, which is the run's last
    confirmed sign of progress. Any other event type (`run_finished`,
    `observation_started`, `stall_detected`, `dataset_diff_completed`, a prior
    `run_stall_detected_*`, ...) is neither a heartbeat nor a step event and is simply
    skipped: it neither advances the count nor resets it, because only wakeups — not
    "things journalled" in general — are what `_RUN_STALL_WAKEUP_THRESHOLD` counts.

    This is computed against the journal as read *before* this wakeup's own new events
    are appended (`cli.py` journals everything from one wakeup together, after deciding
    what is new), so the count always describes silence strictly prior to the current
    wakeup — consistent with how every other idempotency check in this package reads
    the pre-wakeup journal.

    Args:
        events: The run's journal, as returned by `Journal.read()`.

    Returns:
        The number of heartbeats since the last step_completed/step_failed event, or
        every heartbeat in the journal if there has never been one yet.
    """
    count = 0
    for event in reversed(events):
        if event.event_type in ('step_completed', 'step_failed'):
            return count
        if is_heartbeat(event):
            count += 1
    return count


def run_stalled(
    run_state: str | None,
    counts: dict[str, int],
    stalls: list[StallVerdict],
    events: list[JournalEvent],
    wakeup_threshold: int = _RUN_STALL_WAKEUP_THRESHOLD,
) -> RunStallVerdict | None:
    """Judge whether the run as a whole — not any single task — has stalled.

    Two distinct, cheap signatures, checked in order of how little they need to fire:

    `'stuck_trigger'` needs only this one snapshot: the run is `running`, `counts` shows
    zero tasks in `_ACTIVE_STATES` *and* zero in `_STUCK_TRIGGER_MOVING_STATES`, yet at
    least one task is still `_PENDING_STATE`. A trigger rule that never fired, or a
    scheduler that has given up, leaves exactly this shape — nothing executing, nothing
    queued to execute it, work still waiting. The `_STUCK_TRIGGER_MOVING_STATES` check is
    what makes "nothing queued to execute it" true rather than aspirational: a task sitting
    in `queued`, `scheduled` or `up_for_retry` is a trigger rule that *did* fire, moving the
    task toward `running` — a run's opening minutes, an ordinary step hand-off, and a bounded
    retry backoff all produce exactly this shape and are not what this signature exists to
    catch (see that constant's docstring). This is checked first because it is strictly
    cheaper (no `events` scan) and because the two signatures are mutually exclusive by
    construction: `'no_progress'` below requires at least one active task, `'stuck_trigger'`
    requires zero, so a given snapshot can never trigger both and there is nothing to
    reconcile between them.

    `'no_progress'` needs history. It fires only once *every* condition holds:
    the run is `running`; at least one task is active (otherwise `'stuck_trigger'` is
    the applicable signature, not this one); every active task is already individually
    flagged by `stalled()` (`len(stalls) >= active`) — an active task `stalled()` has
    not yet flagged is still within its own acceptable threshold, and that is exactly
    what "a legitimately long step" looks like, so its presence alone silences this
    rule regardless of how long the run has otherwise been quiet; and
    `_wakeups_since_step_event(events)` has reached `wakeup_threshold`.

    That "every active task already flagged" condition is also how this avoids
    double-reporting the same underlying problem as `stalled()`: it deliberately never
    fires *instead of* a per-task verdict, only *in addition to* one, once every active
    task has already been individually called out. The information this adds beyond
    those per-task bullets is aggregate, not duplicate — "this is not one flaky task,
    the run's *entire* active set is stuck" — which is the one thing no single
    `StallVerdict` can say on its own.

    Args:
        run_state: `Snapshot.run_state`. Only `'running'` can stall this way; a run
            that has not yet started or has already reached a terminal state has
            nothing here to judge.
        counts: `Snapshot.counts` — task instances by state, `_PENDING_STATE` for one
            with none.
        stalls: `Snapshot.stalls` — this wakeup's per-task verdicts from `stalled()`.
        events: The run's journal, as returned by `Journal.read()`, read for
            `_wakeups_since_step_event`.
        wakeup_threshold: Consecutive silent wakeups required for `'no_progress'`.

    Returns:
        A verdict, or None if the run is not `running` or neither signature's
        conditions are fully met. Neither signature catches a scheduler dying while
        tasks sit in `_STUCK_TRIGGER_MOVING_STATES` with nothing `_PENDING_STATE` left
        behind them (e.g. `{'queued': 5, 'success': 127}`): `'stuck_trigger'` declines
        because `moving > 0`, `'no_progress'` because `active == 0`. That is the same
        pre-start-freeze gap the module docstring describes for `stalled()` — deferred
        to a rule of its own, not stretched to fit here.
    """
    if run_state != 'running':
        return None

    active = sum(counts.get(state, 0) for state in _ACTIVE_STATES)
    moving = sum(counts.get(state, 0) for state in _STUCK_TRIGGER_MOVING_STATES)
    pending = counts.get(_PENDING_STATE, 0)

    if active == 0:
        if moving > 0 or pending == 0:
            return None
        return RunStallVerdict(reason='stuck_trigger', pending=pending)

    if len(stalls) < active:
        return None

    wakeups = _wakeups_since_step_event(events)
    if wakeups < wakeup_threshold:
        return None
    return RunStallVerdict(reason='no_progress', wakeups=wakeups, active_tasks=active)


def describe_run_stall(verdict: RunStallVerdict) -> str:
    """Render a `RunStallVerdict` as one line of human-readable text.

    Shared by `snapshot.render_snapshot` (a human pulling a snapshot on demand) and
    `report.render_comment` (the wakeup's GitHub comment), so the two never drift into
    describing the same verdict differently.

    Args:
        verdict: The verdict to describe.

    Returns:
        One line, with no leading label and no trailing punctuation, naming the
        signature that fired and the figures behind it.
    """
    if verdict.reason == 'stuck_trigger':
        return (
            f'no task is active (running/deferred/restarting) while {verdict.pending} '
            'task(s) remain pending — a trigger rule that never fired, or a scheduler '
            'that has given up'
        )
    return (
        f'no step has completed or failed in the last {verdict.wakeups} wakeups, and all '
        f'{verdict.active_tasks} currently active task(s) are already individually '
        'flagged as stalled'
    )
