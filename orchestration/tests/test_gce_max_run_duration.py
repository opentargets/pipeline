"""Tests for step VMs that Compute Engine deletes on its own."""

import datetime
from unittest.mock import MagicMock, patch

import pytest
from google.api_core.exceptions import NotFound

from orchestration.operators.gce import (
    DEFAULT_MAX_RUN_DURATION,
    ComputeEngineRunContainerizedWorkloadSensor,
    DeleteInstanceOperator,
)


def _sensor(**kwargs) -> ComputeEngineRunContainerizedWorkloadSensor:
    return ComputeEngineRunContainerizedWorkloadSensor(
        task_id='a_step',
        instance_name='a-machine',
        container_image='an-image',
        **kwargs,
    )


def test_the_instance_is_deleted_after_the_default_run_duration() -> None:
    scheduling = _sensor().declare_instance().scheduling
    assert scheduling.instance_termination_action == 'DELETE'
    assert scheduling.max_run_duration.seconds == DEFAULT_MAX_RUN_DURATION.total_seconds()


def test_the_run_duration_can_be_set_per_step() -> None:
    scheduling = _sensor(max_run_duration=datetime.timedelta(hours=2)).declare_instance().scheduling
    assert scheduling.max_run_duration.seconds == 7200


@pytest.mark.parametrize(
    ('execution_timeout', 'expected'),
    [
        (None, datetime.timedelta(hours=3)),
        (datetime.timedelta(hours=1), datetime.timedelta(hours=1)),
    ],
)
def test_the_deferral_is_bounded(execution_timeout, expected) -> None:
    sensor = _sensor(max_run_duration=datetime.timedelta(hours=3), execution_timeout=execution_timeout, deferrable=True)
    with (
        patch.object(type(sensor), 'hook'),
        patch.object(sensor, 'start'),
        patch.object(sensor, 'defer') as defer,
    ):
        sensor.execute({'dag_run': MagicMock(run_id='a-run')})
    assert defer.call_args.kwargs['timeout'] == expected


def test_deleting_an_instance_that_is_already_gone_succeeds() -> None:
    op = DeleteInstanceOperator(task_id='delete_vm_a_step', resource_id='a-machine')
    with patch(
        'airflow.providers.google.cloud.operators.compute.ComputeEngineDeleteInstanceOperator.execute',
        side_effect=NotFound('gone'),
    ):
        op.execute({})
