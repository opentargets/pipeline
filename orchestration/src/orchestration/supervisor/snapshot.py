"""One read-only view of a pipeline run, assembled for an agent to judge.

The supervisor is stateless, so this is what it sees on waking: the run's state,
what each task is doing, anything that looks stalled, and how much the journal
already knows. Nothing here writes.

The journal is keyed on the Airflow `dag_run_id` (`--run`), which is the only run
identifier this module has. The design spec's journal path is instead keyed on
`run_name` from `unified_pipeline.yaml` (e.g. `ds/target_refactor`) — an independent
identifier for the same run that this CLI cannot derive `dag_run_id` from or vice
versa. Phase 1 stays self-consistent by keying on `dag_run_id` throughout, but phase
2's `diff_vs_reference` DAG task knows only `run_name`. If it writes to a
`run_name`-keyed prefix, it and the agent will silently keep two separate journals
for what is really one run. This is flagged, not resolved, here.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from orchestration.supervisor.stall import StallVerdict, baseline_from_journal, stalled

_PENDING = 'pending'
"""What a task instance with no state is counted as."""


class Snapshot(BaseModel):
    """Everything the supervisor needs from one wakeup.

    Args:
        dag_id: The DAG.
        run_id: The run.
        taken_at: When the snapshot was taken.
        run_state: Airflow's state for the run.
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
        journal_events: How many events the journal already holds, so the agent can
            tell a first wakeup from a resumption.
    """

    dag_id: str
    run_id: str
    taken_at: datetime
    run_state: str | None
    counts: dict[str, int]
    running: list[str]
    failed: list[str]
    succeeded: list[str]
    durations: dict[str, float]
    try_numbers: dict[str, int] = Field(default_factory=dict)
    stalls: list[StallVerdict]
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

    return Snapshot(
        dag_id=dag_id,
        run_id=run_id,
        taken_at=now,
        run_state=run.state,
        counts=dict(Counter(t.state or _PENDING for t in tasks)),
        running=[t.ref for t in tasks if t.state == 'running'],
        failed=[t.ref for t in tasks if t.state == 'failed'],
        succeeded=[t.ref for t in tasks if t.state == 'success'],
        durations={t.ref: t.duration for t in tasks if t.duration is not None},
        try_numbers={t.ref: t.try_number for t in tasks},
        stalls=verdicts,
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
    lines.append(' '.join(f'{state}={count}' for state, count in sorted(snapshot.counts.items())))

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
