"""An append-only event log for one pipeline run, stored in GCS.

The supervisor is stateless: it re-derives everything on each wakeup, so it needs a
durable record of what it has already observed, reported and done. That record also
outlives the Airflow metadata database, which is destroyed with the VM, making the
journal the only lasting source of per-step durations.

Each event is its own object rather than a line in one file. GCS objects are
immutable, so appending to a single file means read-modify-write, and this journal
has two writers — the reference-diff DAG task and the agent. A lost update would
silently drop the record the journal exists to keep.

Both writers must agree on the object prefix, and that is not settled yet. The agent
(`snapshot.py`) keys it on the Airflow `dag_run_id`, the only run identifier it has.
The design spec's journal path is instead keyed on `run_name` from
`unified_pipeline.yaml` — an independent identifier for the same run. If
`diff_vs_reference` writes to a `run_name`-keyed prefix while the agent reads and
writes a `dag_run_id`-keyed one, the two writers will silently keep separate
journals for what is really one run — the exact failure two writers on one journal
is meant to avoid. See `snapshot.py`'s module docstring for the fuller account.
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

        `map_index` joins the key only when it names a real mapped instance (neither
        None nor -1, Airflow's value for a task instance outside a mapped operator).
        Without that guard, every mapped step's N task instances — sharing one
        `task_id` and therefore one `step` — would collapse onto a single key: the
        first instance's event would record, and `Journal.append` would then silently
        drop every other instance's event as a duplicate of it. Guarding the common,
        unmapped case also means this key is unchanged in shape for every event
        written before `map_index` existed.

        Returns:
            The key, joining event type, step, try number and map index where present.
        """
        parts = [self.event_type]
        if self.step is not None:
            parts.append(self.step)
        if self.try_number is not None:
            parts.append(str(self.try_number))
        if self.map_index is not None and self.map_index != -1:
            parts.append(str(self.map_index))
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
