"""Tests for the supervisor's Airflow REST client."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from orchestration.supervisor.airflow import AirflowClient, DagRun, TaskInstance, token_request


class TestTokenRequest:
    def test_posts_credentials_to_the_fab_token_endpoint(self) -> None:
        """The token endpoint lives in the FAB provider, not in Airflow core.

        Core's /auth/login is a browser redirect and cannot be used headlessly.
        """
        url, body = token_request('http://airflow:8080', 'airflow', 'airflow')
        assert url == 'http://airflow:8080/auth/token'
        assert body == {'username': 'airflow', 'password': 'airflow'}

    def test_strips_a_trailing_slash_from_the_base_url(self) -> None:
        url, _ = token_request('http://airflow:8080/', 'u', 'p')
        assert url == 'http://airflow:8080/auth/token'


class TestTaskInstance:
    def test_parses_a_running_task(self) -> None:
        ti = TaskInstance.model_validate({
            'task_id': 'run_pts_target',
            'state': 'running',
            'try_number': 1,
            'max_tries': 0,
            'duration': None,
            'start_date': '2026-07-21T14:00:00Z',
            'end_date': None,
            'queued_dttm': '2026-07-21T13:58:00Z',
            'operator': 'SubmitJobOperator',
        })
        assert ti.state == 'running'
        assert ti.end_date is None

    def test_accepts_a_null_state(self) -> None:
        """Airflow returns a null state for a task instance that has not been scheduled."""
        ti = TaskInstance.model_validate({
            'task_id': 't', 'state': None, 'try_number': 0, 'max_tries': 0,
            'duration': None, 'start_date': None, 'end_date': None,
            'queued_dttm': None, 'operator': None,
        })
        assert ti.state is None

    def test_ignores_fields_the_supervisor_does_not_use(self) -> None:
        """TaskInstanceResponse carries about thirty fields. Extra keys must not raise."""
        ti = TaskInstance.model_validate({
            'task_id': 't', 'state': 'success', 'try_number': 1, 'max_tries': 0,
            'duration': 12.5, 'start_date': None, 'end_date': None,
            'queued_dttm': None, 'operator': None,
            'pool': 'default_pool', 'hostname': 'x', 'rendered_fields': {},
        })
        assert ti.duration == 12.5

    def test_task_id_is_required(self) -> None:
        with pytest.raises(ValidationError):
            TaskInstance.model_validate({'state': 'success'})

    def test_map_index_defaults_to_unmapped(self) -> None:
        """-1 is Airflow's own value for a task instance that is not part of a mapped operator."""
        assert TaskInstance(task_id='t').map_index == -1

    def test_ref_is_the_bare_task_id_when_unmapped(self) -> None:
        assert TaskInstance(task_id='t', map_index=-1).ref == 't'

    def test_ref_is_qualified_by_map_index_when_mapped(self) -> None:
        assert TaskInstance(task_id='t', map_index=3).ref == 't[3]'

    def test_two_mapped_instances_of_the_same_task_have_distinct_refs(self) -> None:
        """N shards of a `.partial().expand()`'d task share one task_id (unified_pipeline.py:476-482)."""
        a = TaskInstance(task_id='run_gentropy_variant_annotation', map_index=0)
        b = TaskInstance(task_id='run_gentropy_variant_annotation', map_index=1)
        assert a.ref != b.ref


class TestDagRun:
    def test_parses_a_run(self) -> None:
        run = DagRun.model_validate({
            'dag_run_id': 'manual__2026-07-21T15:07:47.545737+00:00',
            'state': 'running',
            'start_date': '2026-07-21T15:07:48Z',
            'end_date': None,
        })
        assert run.state == 'running'
        assert run.start_date == datetime(2026, 7, 21, 15, 7, 48, tzinfo=UTC)


def _response(payload: object, status: int = 200) -> MagicMock:
    r = MagicMock()
    r.status_code = status
    r.json.return_value = payload
    return r


def _session(*responses: MagicMock) -> MagicMock:
    s = MagicMock()
    s.post.return_value = responses[0]
    s.get.side_effect = list(responses[1:])
    return s


def _token() -> MagicMock:
    """A fresh 201 token response per test, so no mock state leaks between them."""
    return _response({'access_token': 'jwt-value'}, status=201)


_CREDENTIAL = 'p'
"""Not a secret.

S106 fires on a string literal passed to a `password` argument, so the value is
hoisted. The name avoids ruff's password/secret name pattern, which is what S105
matches on — it does not trace the value to its use site.
"""


class TestAirflowClient:
    def test_authenticates_once_and_reuses_the_token(self) -> None:
        session = _session(_token(), _response({'task_instances': [], 'total_entries': 0}),
                           _response({'task_instances': [], 'total_entries': 0}))
        client = AirflowClient(session=session, base_url='http://a:8080', username='u', password=_CREDENTIAL)
        client.task_instances('unified_pipeline', 'r')
        client.task_instances('unified_pipeline', 'r')
        assert session.post.call_count == 1

    def test_sends_the_token_as_a_bearer_header(self) -> None:
        session = _session(_token(), _response({'task_instances': [], 'total_entries': 0}))
        client = AirflowClient(session=session, base_url='http://a:8080', username='u', password=_CREDENTIAL)
        client.task_instances('unified_pipeline', 'r')
        assert session.get.call_args.kwargs['headers']['Authorization'] == 'Bearer jwt-value'

    def test_maps_task_instances(self) -> None:
        payload = {
            'task_instances': [{
                'task_id': 'run_pts_target', 'state': 'running', 'try_number': 1,
                'max_tries': 0, 'duration': None, 'start_date': '2026-07-21T14:00:00Z',
                'end_date': None, 'queued_dttm': None, 'operator': 'SubmitJobOperator',
            }],
            'total_entries': 1,
        }
        session = _session(_token(), _response(payload))
        client = AirflowClient(session=session, base_url='http://a:8080', username='u', password=_CREDENTIAL)
        tis = client.task_instances('unified_pipeline', 'r')
        assert [t.task_id for t in tis] == ['run_pts_target']

    def test_requests_the_documented_task_instances_path(self) -> None:
        session = _session(_token(), _response({'task_instances': [], 'total_entries': 0}))
        client = AirflowClient(session=session, base_url='http://a:8080', username='u', password=_CREDENTIAL)
        client.task_instances('unified_pipeline', 'my-run')
        url = session.get.call_args.args[0]
        assert url == 'http://a:8080/api/v2/dags/unified_pipeline/dagRuns/my-run/taskInstances'

    def test_requests_a_stable_sort_order_for_pagination(self) -> None:
        """The endpoint's default sort is map_index, a total tie for every unmapped task.

        Two offset-paginated requests sorted on a tied key are not guaranteed to see
        rows in the same order, so a page boundary can drop a row or return it twice.
        `order_by=id` pins a unique, immutable sort key instead.
        """
        session = _session(_token(), _response({'task_instances': [], 'total_entries': 0}))
        client = AirflowClient(session=session, base_url='http://a:8080', username='u', password=_CREDENTIAL)
        client.task_instances('unified_pipeline', 'r')
        assert session.get.call_args.kwargs['params']['order_by'] == 'id'

    def test_pages_until_every_task_instance_is_collected(self) -> None:
        """The unified pipeline has roughly 150 steps and many tasks per step.

        A single page would silently truncate the run.
        """
        page = {'task_instances': [{'task_id': f't{i}'} for i in range(100)], 'total_entries': 150}
        rest = {'task_instances': [{'task_id': f't{i}'} for i in range(100, 150)], 'total_entries': 150}
        session = _session(_token(), _response(page), _response(rest))
        client = AirflowClient(session=session, base_url='http://a:8080', username='u', password=_CREDENTIAL)
        assert len(client.task_instances('unified_pipeline', 'r')) == 150

    def test_maps_a_dag_run(self) -> None:
        payload = {'dag_run_id': 'r', 'state': 'running', 'start_date': '2026-07-21T15:00:00Z',
                   'end_date': None}
        session = _session(_token(), _response(payload))
        client = AirflowClient(session=session, base_url='http://a:8080', username='u', password=_CREDENTIAL)
        assert client.dag_run('unified_pipeline', 'r').state == 'running'

    def test_a_failed_token_exchange_raises_with_the_status(self) -> None:
        session = _session(_response({'detail': 'bad'}, status=401))
        client = AirflowClient(session=session, base_url='http://a:8080', username='u', password=_CREDENTIAL)
        with pytest.raises(RuntimeError, match='401'):
            client.task_instances('unified_pipeline', 'r')

    def test_a_404_from_dag_run_raises_with_the_status_and_path_rather_than_validating(self) -> None:
        """A typo'd run id must not surface as a ValidationError about a missing dag_run_id.

        That error points the reader at the wrong thing entirely: the response never had
        a `dag_run_id` to validate because the run does not exist.
        """
        session = _session(_token(), _response({'detail': 'DAG Run not found'}, status=404))
        client = AirflowClient(session=session, base_url='http://a:8080', username='u', password=_CREDENTIAL)
        with pytest.raises(RuntimeError, match='404'):
            client.dag_run('unified_pipeline', 'typo')

    def test_a_404_from_dag_run_names_the_path(self) -> None:
        session = _session(_token(), _response({'detail': 'DAG Run not found'}, status=404))
        client = AirflowClient(session=session, base_url='http://a:8080', username='u', password=_CREDENTIAL)
        with pytest.raises(RuntimeError, match='/api/v2/dags/unified_pipeline/dagRuns/typo'):
            client.dag_run('unified_pipeline', 'typo')

    def test_a_500_mid_pagination_raises_rather_than_a_keyerror(self) -> None:
        """A struggling server must not surface as a bare KeyError from the paging loop."""
        page = {'task_instances': [{'task_id': f't{i}'} for i in range(100)], 'total_entries': 150}
        session = _session(_token(), _response(page), _response({'detail': 'server error'}, status=500))
        client = AirflowClient(session=session, base_url='http://a:8080', username='u', password=_CREDENTIAL)
        with pytest.raises(RuntimeError, match='500'):
            client.task_instances('unified_pipeline', 'r')


def _dag_run(dag_run_id: str, state: str, start_date: str | None) -> dict[str, object]:
    return {'dag_run_id': dag_run_id, 'state': state, 'start_date': start_date, 'end_date': None}


class TestActiveDagRun:
    def test_a_single_running_run_is_returned(self) -> None:
        payload = {'dag_runs': [_dag_run('r1', 'running', '2026-07-21T14:00:00Z')]}
        session = _session(_token(), _response(payload))
        client = AirflowClient(session=session, base_url='http://a:8080', username='u', password=_CREDENTIAL)
        run = client.active_dag_run('unified_pipeline')
        assert run is not None
        assert run.dag_run_id == 'r1'

    def test_no_running_run_returns_none_rather_than_raising(self) -> None:
        """The idle pipeline is the ordinary case, not an error."""
        session = _session(_token(), _response({'dag_runs': []}))
        client = AirflowClient(session=session, base_url='http://a:8080', username='u', password=_CREDENTIAL)
        assert client.active_dag_run('unified_pipeline') is None

    def test_several_running_runs_returns_the_newest_by_start_date(self) -> None:
        """The server is asked to sort newest-first; the client trusts and takes the first."""
        payload = {'dag_runs': [
            _dag_run('newer', 'running', '2026-07-21T15:00:00Z'),
            _dag_run('older', 'running', '2026-07-21T14:00:00Z'),
        ]}
        session = _session(_token(), _response(payload))
        client = AirflowClient(session=session, base_url='http://a:8080', username='u', password=_CREDENTIAL)
        run = client.active_dag_run('unified_pipeline')
        assert run is not None
        assert run.dag_run_id == 'newer'

    def test_a_queued_run_is_not_mistaken_for_active(self) -> None:
        """Queued runs have not started; the client re-checks state rather than trusting the response."""
        payload = {'dag_runs': [_dag_run('queued-one', 'queued', None)]}
        session = _session(_token(), _response(payload))
        client = AirflowClient(session=session, base_url='http://a:8080', username='u', password=_CREDENTIAL)
        assert client.active_dag_run('unified_pipeline') is None

    def test_a_queued_run_ranked_ahead_does_not_crowd_out_the_real_running_run(self) -> None:
        """A queued run's null start_date can sort first under Postgres' NULLS FIRST default for DESC."""
        payload = {'dag_runs': [
            _dag_run('queued-one', 'queued', None),
            _dag_run('running-one', 'running', '2026-07-21T14:00:00Z'),
        ]}
        session = _session(_token(), _response(payload))
        client = AirflowClient(session=session, base_url='http://a:8080', username='u', password=_CREDENTIAL)
        run = client.active_dag_run('unified_pipeline')
        assert run is not None
        assert run.dag_run_id == 'running-one'

    def test_requests_the_running_state_filter(self) -> None:
        session = _session(_token(), _response({'dag_runs': []}))
        client = AirflowClient(session=session, base_url='http://a:8080', username='u', password=_CREDENTIAL)
        client.active_dag_run('unified_pipeline')
        assert session.get.call_args.kwargs['params']['state'] == 'running'

    def test_requests_a_total_order_sort(self) -> None:
        """`start_date` alone can tie between two runs started in the same instant."""
        session = _session(_token(), _response({'dag_runs': []}))
        client = AirflowClient(session=session, base_url='http://a:8080', username='u', password=_CREDENTIAL)
        client.active_dag_run('unified_pipeline')
        assert session.get.call_args.kwargs['params']['order_by'] == ['-start_date', '-id']

    def test_requests_the_documented_dag_runs_path(self) -> None:
        session = _session(_token(), _response({'dag_runs': []}))
        client = AirflowClient(session=session, base_url='http://a:8080', username='u', password=_CREDENTIAL)
        client.active_dag_run('unified_pipeline')
        url = session.get.call_args.args[0]
        assert url == 'http://a:8080/api/v2/dags/unified_pipeline/dagRuns'
