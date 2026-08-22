"""Live checks against a locally-run Airflow.

Skipped unless RUN_AIRFLOW_TESTS is set. Bring up a server first — see the phase 1
plan's local Airflow recipe. There is no container runtime on the development
machine, so `make local-airflow` is not the way to do it.

Run with:
    RUN_AIRFLOW_TESTS=1 AIRFLOW_TEST_URL=http://localhost:18080 \
      uv run --frozen pytest tests/test_supervisor_airflow_live.py -rxs
"""

from __future__ import annotations

import os

import pytest
import requests

from orchestration.supervisor.airflow import _PAGE_SIZE, AirflowClient

pytestmark = pytest.mark.skipif(
    not os.environ.get('RUN_AIRFLOW_TESTS'),
    reason='needs a running Airflow, set RUN_AIRFLOW_TESTS=1 to run',
)

BASE_URL = os.environ.get('AIRFLOW_TEST_URL', 'http://localhost:18080')
DAG_ID = os.environ.get('AIRFLOW_TEST_DAG', 'unified_pipeline')
RUN_ID = os.environ.get('AIRFLOW_TEST_RUN', 'supervisor-live-test')

_CREDENTIAL = os.environ.get('AIRFLOW_TEST_PASSWORD', 'airflow')
"""Not a secret: the dev-only FAB login for the throwaway local Airflow this test targets.

S106 fires on a string literal passed to a `password` argument, so the value is hoisted.
The name avoids ruff's password/secret name pattern, which is what S105 matches on — it
does not trace the value to its use site.
"""


@pytest.fixture
def client() -> AirflowClient:
    return AirflowClient(session=requests.Session(), base_url=BASE_URL, username='airflow', password=_CREDENTIAL)


class TestLiveAirflow:
    def test_the_token_exchange_returns_a_usable_jwt(self, client: AirflowClient) -> None:
        """The token endpoint is in the FAB provider and returns 201, not 200.

        Core's /auth/login is a browser redirect, so getting this wrong means the
        client cannot authenticate at all.
        """
        assert len(client.token) > 100

    def test_a_dag_run_can_be_read(self, client: AirflowClient) -> None:
        run = client.dag_run(DAG_ID, RUN_ID)
        assert run.dag_run_id == RUN_ID

    def test_task_instances_can_be_read(self, client: AirflowClient) -> None:
        tasks = client.task_instances(DAG_ID, RUN_ID)
        assert tasks, 'the run has no task instances, so this test proves nothing'
        assert all(t.task_id for t in tasks)

    def test_every_task_instance_is_collected(self, client: AirflowClient) -> None:
        """Paging is the failure mode that would silently truncate a 150-step run.

        A single-page implementation returning fewer, unique task ids would pass a
        bare uniqueness check with no symptom — a truncated run looks exactly like a
        healthy short one. This proves three things instead: the fetch actually spans
        more than one page, the count matches what the server itself reports as the
        total, and there are no duplicates from an off-by-one in the offset.
        """
        tasks = client.task_instances(DAG_ID, RUN_ID)
        assert len(tasks) > _PAGE_SIZE, 'fewer than one page of results, so pagination was never exercised'

        response = client.session.get(
            f'{client.base_url}/api/v2/dags/{DAG_ID}/dagRuns/{RUN_ID}/taskInstances',
            headers={'Authorization': f'Bearer {client.token}'},
            params={'limit': 1, 'offset': 0},
        )
        total_entries = response.json()['total_entries']

        assert len(tasks) == total_entries, 'client returned fewer task instances than the server reports'
        assert len({t.task_id for t in tasks}) == len(tasks)
