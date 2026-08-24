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
from typing import Any, Literal

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
        task_id: The task's id within the DAG. Shared by every instance of a mapped
            task (`.partial().expand()`) — see `map_index` and `ref`.
        state: Airflow's state, or None for a task instance not yet scheduled.
        try_number: Which attempt this is.
        max_tries: Retries configured for the task. Zero throughout this pipeline.
        duration: Seconds from start to end, or None while running.
        start_date: When the task instance started executing.
        end_date: When it finished, or None while running.
        queued_dttm: When it was queued. The gap to `start_date` is queueing time,
            which is part of task wall time but is not execution.
        operator: The operator class name, useful for telling GCE steps from Dataproc.
        map_index: Which instance this is among a mapped task's expansion, or -1 for a
            task instance that is not part of one — see `ref`.
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
    map_index: int = -1

    @property
    def ref(self) -> str:
        """This task instance's identity, qualified with its map_index when it has one.

        The two Google Batch steps use `.partial(task_id=...).expand(...)`
        (`dags/unified_pipeline.py:476-482`), so N task instances share one `task_id`
        and it alone cannot tell them apart. -1 is Airflow's value for every task
        instance that is not part of a mapped operator — the overwhelming majority — so
        it is left off entirely rather than appended as `[-1]`, keeping this identical
        to `task_id` for every caller that never sees a mapped task.

        Returns:
            `task_id`, or `task_id[map_index]` when `map_index` is not -1.
        """
        return self.task_id if self.map_index == -1 else f'{self.task_id}[{self.map_index}]'


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


_PAGE_SIZE = 100
"""Task instances per request. The unified pipeline has 132 steps."""


class AirflowClient:
    """Reads DAG runs and task instances from the Airflow REST API.

    Args:
        session: A `requests.Session`, injected so no unit test needs a server.
        base_url: The API server's base URL.
        username: FAB username.
        password: FAB password.
    """

    def __init__(self, session: Any, base_url: str, username: str, password: str) -> None:
        self.session = session
        self.base_url = base_url.rstrip('/')
        self.username = username
        self.password = password
        self._token: str | None = None

    @property
    def token(self) -> str:
        """The bearer token, fetched once and cached for this client's lifetime.

        Returns:
            The JWT.

        Raises:
            RuntimeError: If the token exchange does not return HTTP 201.
        """
        if self._token is None:
            url, body = token_request(self.base_url, self.username, self.password)
            response = self.session.post(url, json=body)
            if response.status_code != 201:
                raise RuntimeError(f'airflow token exchange failed with HTTP {response.status_code}')
            self._token = response.json()['access_token']
        return self._token

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Issue one GET and return its decoded body.

        Args:
            path: The API path, relative to `base_url`.
            params: Query parameters, if any.

        Returns:
            The decoded JSON body.

        Raises:
            RuntimeError: If the response is not HTTP 200. A non-2xx body — a 404 for a
                typo'd run id, a 403 for bad credentials, a 500 from a struggling server —
                does not have the shape callers expect, and letting it reach
                `DagRun.model_validate` or the paging loop below surfaces a `ValidationError`
                about a missing field, or a `KeyError`, that points at the wrong thing
                entirely. This mirrors the token exchange's own status check above.
        """
        response = self.session.get(
            f'{self.base_url}{path}',
            headers={'Authorization': f'Bearer {self.token}'},
            params=params or {},
        )
        if response.status_code != 200:
            raise RuntimeError(f'airflow API returned HTTP {response.status_code} for {path}')
        return response.json()

    def dag_run(self, dag_id: str, run_id: str) -> DagRun:
        """Read one DAG run.

        Args:
            dag_id: The DAG's id.
            run_id: The run's id.

        Returns:
            The run.
        """
        return DagRun.model_validate(self._get(f'/api/v2/dags/{dag_id}/dagRuns/{run_id}'))

    def task_instances(self, dag_id: str, run_id: str) -> list[TaskInstance]:
        """Read every task instance in one DAG run, following pagination.

        The request pins `order_by=id` explicitly. The endpoint's default sort is
        `map_index`, which is a total tie across offset-paginated requests: every task
        in this pipeline has `map_index=-1` except the one mapped group. Two pages are
        two separate queries, and Postgres does not guarantee seq-scan order is stable
        between them once concurrent writes are moving rows around a live run — a row
        can come back on both pages while another never appears on either. `id` is the
        task instance's uuid7 primary key, unique and immutable, which is what offset
        pagination actually requires to be correct rather than merely usually correct.

        Args:
            dag_id: The DAG's id.
            run_id: The run's id.

        Returns:
            Every task instance, ordered by id.
        """
        path = f'/api/v2/dags/{dag_id}/dagRuns/{run_id}/taskInstances'
        collected: list[TaskInstance] = []
        while True:
            page = self._get(path, {'limit': _PAGE_SIZE, 'offset': len(collected), 'order_by': 'id'})
            collected.extend(TaskInstance.model_validate(t) for t in page['task_instances'])
            if len(collected) >= page['total_entries'] or not page['task_instances']:
                return collected
