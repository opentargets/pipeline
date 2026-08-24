"""One read-only view of a pipeline run, assembled for an agent to judge.

The supervisor is stateless, so this is what it sees on waking: the run's state,
what each task is doing, anything that looks stalled, and how much the journal
already knows. Nothing here writes.

**The journal is keyed on the Airflow `dag_run_id` (`--run`), not `run_name` from
`unified_pipeline.yaml`.** The two are independent identifiers for the same run,
and this CLI cannot derive one from the other. `dag_run_id` was chosen because it
comes from Airflow, is authoritative for the run in flight, and needs no file read
to obtain. The prefix is `_agent/{dag_id}/{dag_run_id}/journal`, built in `cli.py`.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from orchestration.supervisor.stall import (
    RunStallVerdict,
    StallVerdict,
    baseline_from_journal,
    describe_run_stall,
    run_stalled,
    stalled,
)

_PENDING = 'pending'
"""What a task instance with no state is counted as."""


class Snapshot(BaseModel):
    """Everything the supervisor needs from one wakeup.

    Args:
        dag_id: The DAG.
        run_id: The run.
        taken_at: When the snapshot was taken.
        run_state: Airflow's state for the run.
        run_started: When the run itself started, from `DagRun.start_date`. `None` only
            for a run so new Airflow has not yet recorded one — in practice
            `active_dag_run`/`most_recent_dag_run` already filter to runs that have
            started, so this is `None` in tests more often than in production.

            This is the third time a `take_snapshot` narrowing has had to be reversed:
            once for `duration`, once for `try_number`, both dropped from `TaskInstance`
            because no caller needed them *yet*, both added back once one did.
            `TaskInstance` earns that piecemeal treatment on its own terms — its
            docstring explains it is a deliberate ~30-field reduction, so hand-picking
            which of many fields to keep is the right shape of fix there. `DagRun` does
            not share that shape: it is four fields total, all cheap, and
            `take_snapshot` already fetches the whole thing on every call — there is
            nothing left to selectively avoid. Keeping the full `run: DagRun` on
            `Snapshot`, rather than hand-copying scalars off it one at a time, would
            close this particular narrowing for good. That refactor was not made here —
            it would touch `render_snapshot`, `observer.py`, `cli.py` and every existing
            test's `run_state=...` construction for a change wider than this task asked
            for — so only `started` is added, since only it has a caller in this branch
            (`report.py`'s run-stall rendering). `end_date` is deliberately left behind
            again: it has no consumer yet, and a field with no test exercising real
            usage is worse than no field. If a fourth `DagRun` scalar is ever needed,
            that is the point to stop and take the `run: DagRun` refactor instead of
            adding a fourth one-off.
        counts: Task instances by state, with stateless ones counted as pending.
        running: Task refs currently running — `task_id`, qualified with `map_index`
            for a mapped task instance (see `TaskInstance.ref`).
        failed: Task refs that have failed, qualified the same way.
        succeeded: Task refs that have finished successfully, qualified the same way.
        durations: Seconds each task instance took, keyed by its ref, wherever
            Airflow reports one (`TaskInstance.duration` is None while running, so a
            task instance not yet finished is simply absent rather than carrying a
            fabricated `None` or `0.0`). Not filtered to `succeeded` — a failed task
            instance has a duration too. This is a plain record of what was seen;
            deciding which durations are worth keeping (a step's own run task,
            successful) is `observer.py`'s job, not this module's.
        try_numbers: Which attempt each task instance is, keyed by its ref, mirroring
            `TaskInstance.try_number`. Unlike `durations` this has no missing case for
            any `Snapshot` `take_snapshot` builds — `try_number` defaults to 0 on
            `TaskInstance`, not None, so every task instance in
            `running`/`failed`/`succeeded`/`stalls` has an entry here. This is what
            lets `observer.py` tell a step that failed, was re-run and failed again
            apart from the failure already reported for it — see
            `JournalEvent.try_number`'s docstring. Defaults to `{}`, unlike its
            siblings above, so a `Snapshot` built by code written before this field
            existed keeps working; `observer.py` degrades gracefully on a missing
            entry (`try_number=None`, the same as before this field was added), it
            does not raise.
        stalls: Running tasks judged to have stalled.
        run_stall: The run as a whole, judged by `stall.run_stalled` from `counts`,
            `stalls` and the journal's heartbeat history — distinct from `stalls`,
            which is per task. `None` when neither of its two signatures fires; see
            `run_stalled` for both.
        journal_events: How many events the journal already holds, so the agent can
            tell a first wakeup from a resumption.
    """

    dag_id: str
    run_id: str
    taken_at: datetime
    run_state: str | None
    run_started: datetime | None = None
    counts: dict[str, int]
    running: list[str]
    failed: list[str]
    succeeded: list[str]
    durations: dict[str, float]
    try_numbers: dict[str, int] = Field(default_factory=dict)
    stalls: list[StallVerdict]
    run_stall: RunStallVerdict | None = None
    journal_events: int


def take_snapshot(client: Any, journal: Any, dag_id: str, run_id: str, now: datetime) -> Snapshot:
    """Assemble a snapshot of one run.

    Args:
        client: An `AirflowClient`.
        journal: The run's `Journal`.
        dag_id: The DAG to read.
        run_id: The run to read.
        now: The current time, injected so stall verdicts are testable.

    Returns:
        The snapshot.
    """
    run = client.dag_run(dag_id, run_id)
    tasks = client.task_instances(dag_id, run_id)
    events = journal.read()
    baseline = baseline_from_journal(events)

    verdicts = [v for v in (stalled(t, baseline, now) for t in tasks) if v is not None]
    counts = dict(Counter(t.state or _PENDING for t in tasks))

    return Snapshot(
        dag_id=dag_id,
        run_id=run_id,
        taken_at=now,
        run_state=run.state,
        run_started=run.start_date,
        counts=counts,
        running=[t.ref for t in tasks if t.state == 'running'],
        failed=[t.ref for t in tasks if t.state == 'failed'],
        succeeded=[t.ref for t in tasks if t.state == 'success'],
        durations={t.ref: t.duration for t in tasks if t.duration is not None},
        try_numbers={t.ref: t.try_number for t in tasks},
        stalls=verdicts,
        run_stall=run_stalled(run.state, counts, verdicts, events),
        journal_events=len(events),
    )


def render_snapshot(snapshot: Snapshot) -> str:
    """Render a snapshot for a human.

    Stalls and failures come first and are never folded into the counts — both are
    escalations, and a digest is where an escalation goes to be ignored.

    Args:
        snapshot: The snapshot to render.

    Returns:
        The rendered text.
    """
    lines = [f'{snapshot.dag_id} / {snapshot.run_id} — {snapshot.run_state or "unknown"}']
    if snapshot.run_started is not None:
        lines.append(f'started {snapshot.run_started.strftime("%Y-%m-%d %H:%M UTC")}')
    lines.append(' '.join(f'{state}={count}' for state, count in sorted(snapshot.counts.items())))

    if snapshot.run_stall is not None:
        lines.append('')
        lines.append(f'RUN STALL: {describe_run_stall(snapshot.run_stall)}')

    if snapshot.stalls:
        lines.append('')
        for verdict in snapshot.stalls:
            hours = verdict.elapsed / 3600
            limit = verdict.threshold / 3600
            lines.append(
                f'STALL {verdict.task_id}: running {hours:.1f}h against a {limit:.1f}h '
                f'threshold ({verdict.basis})'
            )

    if snapshot.failed:
        lines.append('')
        lines.append('failed: ' + ', '.join(snapshot.failed))

    return '\n'.join(lines)
