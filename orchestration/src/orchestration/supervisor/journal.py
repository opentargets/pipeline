"""An append-only event log for one pipeline run, stored in GCS.

The supervisor is stateless: it re-derives everything on each wakeup, so it needs a
durable record of what it has already observed, reported and done. That record also
outlives the Airflow metadata database, which is destroyed with the VM, making the
journal the only lasting source of per-step durations.

Each event is its own object rather than a line in one file. GCS objects are
immutable, so appending to a single file means read-modify-write, and this journal
has two writers — the reference-diff DAG task and the agent. A lost update would
silently drop the record the journal exists to keep.

**The journal is keyed on the Airflow `dag_run_id`, not `run_name`.** Both would
have identified the same run, but `dag_run_id` comes from Airflow, is authoritative
for the run in flight, and needs no file read to obtain — see `snapshot.py`'s module
docstring for the fuller account. The `diff_vs_reference` DAG task this module's
docstring used to warn about is superseded: the observer diffs once at terminal
state instead (see `cli.py`'s module docstring), so there is now only one writer,
and it always keys on `dag_run_id`.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator


class JournalEvent(BaseModel):
    """One thing that happened during a run.

    Args:
        event_type: What happened, e.g. `step_completed`, `step_failed`,
            `stall_detected`, `retrigger`.
        step: The step it happened to, or None for run-level events. This is always the
            bare `unified_pipeline.yaml` step name (`pts_target`), never the qualified
            Airflow `task_id` (`pts_target.run_pts_target`) — the same spelling GCP uses
            as its billing label and `usage.StepUsage.step` already carries. See
            `stall.baseline_from_journal` for why the two spellings must not be mixed.
        try_number: Which attempt, so a retry is a distinct event.
        map_index: Which instance of a mapped task this happened to, mirroring
            `TaskInstance.map_index` — -1 (or None, for an event with no task instance
            behind it) for one that is not part of a mapped operator. See `key` for why
            it is folded into the key only when it distinguishes anything.
        at: When the agent observed it.
        payload: Event-specific detail. Durations live here, which is what makes the
            journal the durable duration record.
    """

    event_type: str
    step: str | None = None
    try_number: int | None = None
    map_index: int | None = None
    at: datetime
    payload: dict[str, Any] = Field(default_factory=dict)

    @field_validator('event_type', 'step')
    @classmethod
    def _forbid_slash(cls, value: str | None) -> str | None:
        """Reject a `/` in a key component.

        A key becomes a path segment in the object name (`{key}/{timestamp}.json`), so a
        `/` inside `event_type` or `step` would let one event's key become a path-prefix
        of another's — the same silent-skip hole the `/`-delimited naming scheme exists
        to close, reopened one level down.
        """
        if value is not None and '/' in value:
            raise ValueError("must not contain '/': it becomes a path segment in the journal object name")
        return value

    @field_validator('step')
    @classmethod
    def _forbid_qualified_task_id(cls, value: str | None) -> str | None:
        """Reject a fully-qualified Airflow `task_id` where a bare step name belongs.

        `step` must be the bare `unified_pipeline.yaml` step name (`pts_target`), never
        the `task_group`-qualified `task_id` (`pts_target.run_pts_target`) — see this
        class's docstring and `stall.baseline_from_journal` for why the two must not be
        mixed. None of `unified_pipeline.yaml`'s step names contains a `.`, so a `.`
        here unambiguously means a caller passed a `task_id`. Catching that at
        construction time turns a permanent, symptomless silent-miss (a baseline keyed
        in a namespace nothing ever looks up, `basis == 'ceiling'` forever, identical to
        an honest first run) into an immediate `ValidationError` at the call site that
        made the mistake.
        """
        if value is not None and '.' in value:
            raise ValueError(
                f"step={value!r} looks like a qualified Airflow task_id, not a step name: "
                "strip everything from the first '.' onward (e.g. via step_from_task_id) "
                'before constructing this event'
            )
        return value

    @property
    def key(self) -> str:
        """The idempotency key.

        Two observations of the same thing produce the same key, which is what stops
        the fourth wakeup re-reporting the first wakeup's completions.

        `try_number` and `map_index` are both optional, and a plain positional join
        of optional parts is ambiguous: `step_failed-pts_target-1` cannot tell
        `try_number=1, map_index=-1` (or None) apart from `try_number=None,
        map_index=1` — both produce the identical trailing `-1`. Each is tagged with
        its own letter instead — `t` for `try_number`, `m` for `map_index` — so
        `step_failed-pts_target-t1` and `step_failed-pts_target-m1` cannot be
        confused with one another, by construction, regardless of which optional
        parts are present or absent. `map_index` still joins only when it names a
        real mapped instance (neither None nor -1, Airflow's value for a task
        instance outside a mapped operator): without that guard, every mapped step's
        N task instances — sharing one `task_id` and therefore one `step` — would
        collapse onto a single key, and `Journal.append` would silently drop every
        instance but the first as a duplicate of it.

        No journal has ever been written in production — `baseline_from_journal` had
        no writer until this branch, and the cron that would run `observe` is not yet
        enabled — so there is no installed base of untagged keys this format needs to
        stay compatible with. That is why the tags were free to add now and this
        docstring no longer explains how to preserve the old shape.

        Returns:
            The key: event type, then step, then `t{try_number}` and `m{map_index}`
            wherever each is present, joined with `-`.
        """
        parts = [self.event_type]
        if self.step is not None:
            parts.append(self.step)
        if self.try_number is not None:
            parts.append(f't{self.try_number}')
        if self.map_index is not None and self.map_index != -1:
            parts.append(f'm{self.map_index}')
        return '-'.join(parts)


class Journal:
    """The append-only event log for one run.

    Args:
        bucket: A GCS bucket, injected so no unit test needs credentials.
        prefix: The object prefix for this run's journal, without a trailing slash.
    """

    def __init__(self, bucket: Any, prefix: str) -> None:
        self.bucket = bucket
        self.prefix = prefix.rstrip('/')

    def _name(self, event: JournalEvent) -> str:
        return f'{self.prefix}/{event.key}/{event.at.isoformat()}.json'

    def _key_prefix(self, key: str) -> str:
        return f'{self.prefix}/{key}/'

    def has(self, key: str) -> bool:
        """Whether an event with this key has been recorded.

        A key is its own path segment, delimited by `/` on both sides, rather than a
        string suffix of the object name. A string-suffix check would mistake a shorter
        key for a match inside a longer one that happens to end the same way — `a-b`
        inside `x-a-b`, for instance — and a false "already recorded" is exactly the
        failure this journal exists to prevent. Delimiting the key on both sides makes
        that structurally impossible, and turns the check into a targeted GCS prefix
        query instead of a scan of every object in the run's journal.

        Args:
            key: The idempotency key to look for.

        Returns:
            True if it is present.
        """
        return next(iter(self.bucket.list_blobs(prefix=self._key_prefix(key))), None) is not None

    def append(self, event: JournalEvent) -> bool:
        """Record an event, unless its key is already present.

        Args:
            event: The event to record.

        Returns:
            True if it was written, False if it was already there.
        """
        if self.has(event.key):
            return False
        self.bucket.blob(self._name(event)).upload_from_string(
            event.model_dump_json(),
            content_type='application/json',
        )
        return True

    def read(self) -> list[JournalEvent]:
        """Every event recorded for this run.

        Object names sort by key first under this layout, not by timestamp, so
        chronological order cannot come from listing order and is applied explicitly.
        The prefix is anchored with a trailing `/`, matching `_key_prefix`, so a sibling
        prefix that merely starts with the same characters (`journal2` next to
        `journal`) is never swept in.

        Returns:
            The events, ordered chronologically by `at`.
        """
        events = [
            JournalEvent.model_validate(json.loads(blob.download_as_text()))
            for blob in self.bucket.list_blobs(prefix=f'{self.prefix}/')
        ]
        return sorted(events, key=lambda event: event.at)


_HEARTBEAT_PREFIX = 'heartbeat_'
"""Every heartbeat event's `event_type` starts with this, followed by a compact UTC
timestamp (see `heartbeat_event`).

`JournalEvent.key` is built from `event_type`/`step`/`try_number`/`map_index` alone —
deliberately not `at` — so that two *observations* of the same real-world thing
collapse onto one key. A heartbeat has no such real-world thing behind it: it exists
solely to prove this exact wakeup happened, so a constant `event_type` (`'heartbeat'`
alone) would produce the identical key on every wakeup and `Journal.append` would
silently no-op every heartbeat after the first — the opposite of what a liveness
marker needs. Folding the timestamp into `event_type` itself (rather than `step`,
which `_forbid_qualified_task_id` reserves for a real step name, or `map_index`/
`try_number`, whose fields have their own meanings) is what makes every heartbeat's
key distinct, using the one key-bearing field a heartbeat can freely repurpose."""


def heartbeat_event(at: datetime) -> JournalEvent:
    """Build one wakeup's heartbeat: proof this run was observed, carrying no verdict.

    A heartbeat is journalled unconditionally, once per wakeup, alongside whatever else
    that wakeup found — see `cli.py`'s module docstring for why this closes the
    liveness gap `_OBSERVATION_STARTED_EVENT` could not (a dead cron and a quiet one
    used to journal the same thing). `stall.run_stalled` also reads these back to count
    consecutive silent wakeups without relying on wall-clock time, which is what makes
    that count robust to the cron itself having been down for a while — see its
    docstring.

    Args:
        at: This wakeup's timestamp, the same instant used for every other event this
            wakeup journals. Assumed UTC, like every other `at` in this package.

    Returns:
        The event. `step`/`try_number`/`map_index` are all left at their `None`
        defaults — a heartbeat is a run-level fact, not about any one step.
    """
    timestamp = at.strftime('%Y%m%dT%H%M%SZ')
    return JournalEvent(event_type=f'{_HEARTBEAT_PREFIX}{timestamp}', at=at)


def is_heartbeat(event: JournalEvent) -> bool:
    """Whether an event is a heartbeat, as built by `heartbeat_event`.

    Args:
        event: The event to classify.

    Returns:
        True if `event_type` carries the heartbeat prefix.
    """
    return event.event_type.startswith(_HEARTBEAT_PREFIX)
