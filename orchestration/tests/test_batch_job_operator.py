"""Tests for label propagation from BatchJobOperator down to the billed Batch resources."""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest

from orchestration.models.batch.environment import EnvironmentRegistrySpec, EnvironmentSpec
from orchestration.models.batch.instance import AllocationSpec, InstanceResourceSpec, InstanceSpec
from orchestration.models.batch.job import JobSpec
from orchestration.models.batch.logs import LogsSpec
from orchestration.models.batch.operator import BatchIndexRow, BatchJobOperatorSpec
from orchestration.models.batch.runnable import RunnableSpec
from orchestration.models.batch.task import TaskConfiguration
from orchestration.models.batch.task_group import TaskGroupSpec
from orchestration.operators.batch.batch_job_operator import BatchJobOperator
from orchestration.utils.labels import Labels

RUN_ID = 'manual__2026-08-21T09:00:00+00:00'
EXPECTED_RUN_LABEL = 'manual__2026-08-21t09-00-00-00-00'


@pytest.fixture
def environments() -> EnvironmentRegistrySpec:
    return EnvironmentRegistrySpec(environments=[EnvironmentSpec(variables={'TASK_INDEX': '0'})])


@pytest.fixture
def job_spec(environments: EnvironmentRegistrySpec) -> JobSpec:
    return JobSpec(
        task_group=TaskGroupSpec(
            parallelism=1,
            task_config=TaskConfiguration(
                instance_resource_spec=InstanceResourceSpec(cpu_milli=1000, memory_mib=2048, boot_disk_mib=51200),
                runnable_spec=RunnableSpec(image_uri='gcr.io/p/img:latest', inline_commands=['echo', 'hi']),
            ),
            task_environments=environments,
        ),
        allocation=AllocationSpec(instance=InstanceSpec()),
        logs=LogsSpec(),
    )


@pytest.fixture
def operator(job_spec: JobSpec, environments: EnvironmentRegistrySpec) -> BatchJobOperator:
    return BatchJobOperator(
        task_id='run_gentropy_l2g_prediction',
        job_name='gentropy-l2g-prediction',
        batch_index_row=BatchIndexRow(idx=0, environments=environments),
        batch_job_spec=BatchJobOperatorSpec(job=job_spec),
        labels=Labels({'tool': 'gentropy', 'step': 'gentropy_l2g_prediction'}),
    )


@pytest.fixture
def context() -> dict:
    return {'dag_run': MagicMock(run_id=RUN_ID), 'params': {}}


def execute(operator: BatchJobOperator, context: dict) -> None:
    """Run the operator's execute with the parent's submit-and-wait stubbed out."""
    target = 'orchestration.operators.batch.batch_job_operator.CloudBatchSubmitJobOperator.execute'
    with patch(target, return_value={}):
        operator.execute(context)  # ty:ignore[invalid-argument-type]


def job_labels(operator: BatchJobOperator) -> dict[str, str]:
    """Labels on the job object itself, which only reach the job and its log entries."""
    return cast('dict[str, Any]', operator.job)['labels']


def allocation_labels(operator: BatchJobOperator) -> dict[str, str]:
    """Labels on the allocation policy, which Batch puts on the billed VMs and disks."""
    return cast('dict[str, Any]', operator.job)['allocation_policy']['labels']


def test_allocation_spec_build_applies_the_label_override() -> None:
    allocation = AllocationSpec(instance=InstanceSpec(), labels={'team': 'from-spec'})

    policy = allocation.build(labels={'team': 'from-operator'})

    assert dict(policy.labels) == {'team': 'from-operator'}


def test_allocation_spec_build_keeps_its_own_labels_without_an_override() -> None:
    allocation = AllocationSpec(instance=InstanceSpec(), labels={'team': 'from-spec'})

    assert dict(allocation.build().labels) == {'team': 'from-spec'}


def test_job_spec_build_forwards_labels_to_the_allocation_policy(job_spec: JobSpec) -> None:
    labels = {'tool': 'gentropy', 'step': 'gentropy_l2g_prediction'}

    job = job_spec.build(labels=labels)

    assert dict(job.labels) == labels
    assert dict(job.allocation_policy.labels) == labels


def test_init_puts_the_static_labels_on_the_allocation_policy(operator: BatchJobOperator) -> None:
    assert allocation_labels(operator)['tool'] == 'gentropy'
    assert allocation_labels(operator)['step'] == 'gentropy_l2g_prediction'


def test_execute_adds_the_run_label_to_the_job(operator: BatchJobOperator, context: dict) -> None:
    execute(operator, context)

    assert job_labels(operator)['run'] == EXPECTED_RUN_LABEL


def test_execute_adds_the_run_label_to_the_allocation_policy(operator: BatchJobOperator, context: dict) -> None:
    execute(operator, context)

    assert allocation_labels(operator)['run'] == EXPECTED_RUN_LABEL


def test_execute_keeps_the_job_and_allocation_labels_identical(operator: BatchJobOperator, context: dict) -> None:
    execute(operator, context)

    assert job_labels(operator) == allocation_labels(operator)


def test_execute_prefers_the_run_label_param_over_the_dag_run_id(operator: BatchJobOperator, context: dict) -> None:
    context['params'] = {'run_label': '26.09'}

    execute(operator, context)

    assert allocation_labels(operator)['run'] == '26-09'


def test_execute_does_not_mutate_the_labels_passed_by_the_dag(
    job_spec: JobSpec,
    environments: EnvironmentRegistrySpec,
    context: dict,
) -> None:
    labels = Labels({'tool': 'gentropy', 'step': 'gentropy_l2g_prediction'})
    operator = BatchJobOperator(
        task_id='run_gentropy_l2g_prediction',
        job_name='gentropy-l2g-prediction',
        batch_index_row=BatchIndexRow(idx=0, environments=environments),
        batch_job_spec=BatchJobOperatorSpec(job=job_spec),
        labels=labels,
    )

    execute(operator, context)

    assert 'run' not in labels


def test_labels_fall_back_to_the_job_spec_when_the_dag_passes_none(
    job_spec: JobSpec,
    environments: EnvironmentRegistrySpec,
    context: dict,
) -> None:
    job_spec.labels = {'tool': 'harmonisation'}
    operator = BatchJobOperator(
        task_id='harmonisation',
        job_name='harmonisation',
        batch_index_row=BatchIndexRow(idx=0, environments=environments),
        batch_job_spec=BatchJobOperatorSpec(job=job_spec),
    )

    execute(operator, context)

    assert allocation_labels(operator)['tool'] == 'harmonisation'
    assert allocation_labels(operator)['run'] == EXPECTED_RUN_LABEL
    assert allocation_labels(operator)['team'] == 'open-targets'
