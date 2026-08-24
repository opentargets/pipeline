"""Deciding what one wakeup has not already reported.

The supervisor is stateless: every wakeup re-derives the run's state from Airflow
(`Snapshot`) and re-reads the journal (`JournalEvent`), rather than remembering
anything between wakeups itself. `observe` is where that re-derivation earns its
name — it is the one place idempotency is decided, by comparing what the snapshot
shows against the `JournalEvent.key`s already on record. Everything downstream (the
rendered comment, the journal write) only ever sees what this function decided is
new; it does no I/O itself.

**A completion is reported only for a step's own run task**, gated by `is_run_task`
— the same gate `stalled` uses before consulting a baseline. A sibling in the same
task group (`delete_vm_pts_target`, a `diff_` task) finishing is not the step
finishing, and its duration is not the step's duration: `stall.baseline_from_journal`
keys the baseline on the bare step name alone, so a sibling's duration journalled
under that key would sit in the same baseline the run task's own duration feeds —
dead weight at best, a fabricated threshold at worst, for a sibling that is not what
`stalled` is timing when it later judges that step. `Snapshot.durations` itself stays
a plain, unfiltered record of everything Airflow reported a duration for; the
filtering happens here.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

from orchestration.supervisor.journal import JournalEvent
from orchestration.supervisor.snapshot import Snapshot
from orchestration.supervisor.step_identity import is_run_task, step_from_task_id

_TERMINAL_RUN_STATES = frozenset({'success', 'failed'})
"""Airflow DAG run states that mean the run itself is over, one way or the other."""

_KEY_TIMESTAMP = datetime(1970, 1, 1, tzinfo=UTC)
"""A throwaway `at` for building a candidate `JournalEvent` purely to read its `key`.

`JournalEvent.key` does not depend on `at`, but `at` is a required field, so
something has to be supplied. This is never journalled anywhere — `observe` writes
nothing — so its value is arbitrary.
"""


def _parse_ref(ref: str) -> tuple[str, int]:
    """Split a task ref back into the task_id and map_index it was built from.

    The inverse of `TaskInstance.ref`: a bare ref carries `map_index=-1`, Airflow's
    value for a task instance outside a mapped operator; a `task_id[N]` ref names the
    N'th instance of a mapped one.

    Args:
        ref: A task ref, as carried by `Snapshot.failed` or `StallVerdict.task_id`.

    Returns:
        The task_id and its map_index.
    """
    if ref.endswith(']') and '[' in ref:
        task_id, _, tail = ref.rpartition('[')
        return task_id, int(tail[:-1])
    return ref, -1


class StepFailure(BaseModel):
    """One failed task instance, not yet reported.

    Args:
        ref: The task's ref — `task_id`, qualified with `map_index` for a mapped
            instance (see `TaskInstance.ref`) — the identity a human needs to find it
            in the Airflow UI, and what tells two failed shards of one mapped step
            apart.
        step: The bare `unified_pipeline.yaml` step name, ready to journal as
            `JournalEvent.step` (which forbids the qualified `task_id` form).
        map_index: Which instance of a mapped task this is, or -1 outside one —
            mirrors `TaskInstance.map_index`, ready to journal as
            `JournalEvent.map_index`.
    """

    ref: str
    step: str
    map_index: int


class StepStall(BaseModel):
    """One running task instance judged to have stalled, not yet reported.

    Args:
        ref: The task's ref, as `StepFailure.ref`.
        step: The bare step name, as `StepFailure.step`.
        map_index: As `StepFailure.map_index`.
        elapsed: Seconds it has been running, from `StallVerdict.elapsed`.
        threshold: The threshold it passed, from `StallVerdict.threshold`.
        basis: Which rule fired — `history` for a step with observed runs (rare in
            practice; see `stall.py`'s module docstring), `ceiling` for one without.
    """

    ref: str
    step: str
    map_index: int
    elapsed: float
    threshold: float
    basis: Literal['history', 'ceiling']


class StepCompletion(BaseModel):
    """One step's own run task, found to have finished successfully, not yet reported.

    Args:
        ref: The task's ref, as `StepFailure.ref`.
        step: The bare step name, as `StepFailure.step`.
        map_index: As `StepFailure.map_index`.
        duration: Seconds it took, from `Snapshot.durations`. This is what
            `stall.baseline_from_journal` reads back out of the `step_completed`
            event this is journalled as, so a completion is only ever reported when a
            duration is actually available for it (see `observe`).
    """

    ref: str
    step: str
    map_index: int
    duration: float


class Observation(BaseModel):
    """What one wakeup has not already reported.

    Every entry here is new relative to the journal `observe` was given: a key
    already present for a failure, a stall, or the run itself means it is absent
    from the corresponding field, not merely deduplicated within it.

    Args:
        failed: Newly failed task instances.
        stalled: Newly stalled task instances.
        completed: Newly completed steps — their own run task, successfully
            finished. Siblings in the same task group are not represented here; see
            the module docstring for why.
        run_finished: The run's terminal state (`success` or `failed`), if the run
            has just reached one and that has not already been reported. None both
            when the run is not yet terminal and when its terminal state was already
            reported on an earlier wakeup — the two read the same from here, since
            both mean there is nothing new to say about the run itself.
    """

    failed: list[StepFailure] = Field(default_factory=list)
    stalled: list[StepStall] = Field(default_factory=list)
    completed: list[StepCompletion] = Field(default_factory=list)
    run_finished: str | None = None

    @property
    def is_empty(self) -> bool:
        """Whether this wakeup found nothing new at all.

        A caller must not infer "nothing new" from any single empty field —
        `run_finished` in particular can carry news on its own, with everything else
        empty.

        Returns:
            True if `failed`, `stalled` and `completed` are all empty and
            `run_finished` is None.
        """
        return not self.failed and not self.stalled and not self.completed and self.run_finished is None


def _already_known(event_type: str, step: str, map_index: int, known: set[str]) -> bool:
    """Whether the key this observation would journal is already on record.

    Builds the candidate event through `JournalEvent` itself, rather than
    replicating `JournalEvent.key`'s join logic here, so the two can never drift
    apart.

    Args:
        event_type: The event type the caller would journal.
        step: The bare step name.
        map_index: The task instance's map_index.
        known: Keys already present in the journal.

    Returns:
        True if a matching event is already recorded.
    """
    candidate = JournalEvent(event_type=event_type, step=step, map_index=map_index, at=_KEY_TIMESTAMP)
    return candidate.key in known


def observe(snapshot: Snapshot, events: list[JournalEvent]) -> Observation:
    """Decide what this wakeup has not already reported.

    Args:
        snapshot: The run's current state.
        events: Everything the journal already holds for this run (as returned by
            `Journal.read()`) — the idempotency record this function checks against.

    Returns:
        What is new since the journal was last written to.
    """
    known = {event.key for event in events}

    failed = []
    for ref in snapshot.failed:
        task_id, map_index = _parse_ref(ref)
        step = step_from_task_id(task_id)
        if not _already_known('step_failed', step, map_index, known):
            failed.append(StepFailure(ref=ref, step=step, map_index=map_index))

    stalled = []
    for verdict in snapshot.stalls:
        task_id, map_index = _parse_ref(verdict.task_id)
        step = step_from_task_id(task_id)
        if not _already_known('stall_detected', step, map_index, known):
            stalled.append(StepStall(
                ref=verdict.task_id, step=step, map_index=map_index,
                elapsed=verdict.elapsed, threshold=verdict.threshold, basis=verdict.basis,
            ))

    completed = []
    for ref in snapshot.succeeded:
        task_id, map_index = _parse_ref(ref)
        if not is_run_task(task_id):
            continue
        duration = snapshot.durations.get(ref)
        if duration is None:
            continue
        step = step_from_task_id(task_id)
        if not _already_known('step_completed', step, map_index, known):
            completed.append(StepCompletion(ref=ref, step=step, map_index=map_index, duration=duration))

    run_finished = None
    if snapshot.run_state in _TERMINAL_RUN_STATES and 'run_finished' not in known:
        run_finished = snapshot.run_state

    return Observation(failed=failed, stalled=stalled, completed=completed, run_finished=run_finished)
