"""Read-only client for the Airflow REST API.

The API is the only way to read Airflow state from the machine that runs it:
`compose.yaml` publishes a port for the API server alone, and the metadata database
is unreachable from the host. Authentication is a JWT obtained from the FAB
provider's token endpoint — Airflow core's `/auth/login` is a browser redirect and
cannot be used headlessly.

Everything here is a GET. Phase 1 is observational and writes nothing.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict

TaskState = Literal[
    'deferred',
    'failed',
    'queued',
    'removed',
    'restarting',
    'running',
    'scheduled',
    'skipped',
    'success',
    'up_for_reschedule',
    'up_for_retry',
    'upstream_failed',
]
"""Task instance states Airflow reports. A task instance may also have no state."""


class TaskInstance(BaseModel):
    """One task instance, reduced to the fields the supervisor reads.

    `TaskInstanceResponse` carries about thirty fields. This model declares the ones
    the supervisor uses and ignores the rest, so an Airflow upgrade that adds a field
    does not break parsing.

    Args:
        task_id: The task's id within the DAG.
        state: Airflow's state, or None for a task instance not yet scheduled.
        try_number: Which attempt this is.
        max_tries: Retries configured for the task. Zero throughout this pipeline.
        duration: Seconds from start to end, or None while running.
        start_date: When the task instance started executing.
        end_date: When it finished, or None while running.
        queued_dttm: When it was queued. The gap to `start_date` is queueing time,
            which is part of task wall time but is not execution.
        operator: The operator class name, useful for telling GCE steps from Dataproc.
    """

    model_config = ConfigDict(extra='ignore')

    task_id: str
    state: TaskState | None = None
    try_number: int = 0
    max_tries: int = 0
    duration: float | None = None
    start_date: datetime | None = None
    end_date: datetime | None = None
    queued_dttm: datetime | None = None
    operator: str | None = None


class DagRun(BaseModel):
    """One DAG run, reduced to the fields the supervisor reads.

    Args:
        dag_run_id: The run id, which is also the source of the `run` GCP label
            after normalisation through `clean_label`.
        state: The run's state.
        start_date: When the run started.
        end_date: When it finished, or None while running.
    """

    model_config = ConfigDict(extra='ignore')

    dag_run_id: str
    state: str | None = None
    start_date: datetime | None = None
    end_date: datetime | None = None


def token_request(base_url: str, username: str, password: str) -> tuple[str, dict[str, str]]:
    """Build the token exchange for the FAB provider's endpoint.

    Args:
        base_url: The API server's base URL.
        username: FAB username. On the dev VM this comes from
            `_AIRFLOW_WWW_USER_USERNAME`, which `compose.yaml` defaults to `airflow`.
        password: FAB password, likewise from `_AIRFLOW_WWW_USER_PASSWORD`.

    Returns:
        The URL to POST to, and the JSON body to send. The response is HTTP 201 with
        an `access_token` field.
    """
    return f'{base_url.rstrip("/")}/auth/token', {'username': username, 'password': password}
