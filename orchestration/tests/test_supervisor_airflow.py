"""Tests for the supervisor's Airflow REST client."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from orchestration.supervisor.airflow import DagRun, TaskInstance, token_request


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
