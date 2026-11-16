"""Tests for the service account a VM runs as."""

from orchestration.operators.gce import ComputeEngineRunContainerizedWorkloadSensor

VM_ACCOUNT = 'a-vm@a-project.iam.gserviceaccount.com'
CHAIN_ACCOUNT = 'chain@a-project.iam.gserviceaccount.com'


def _sensor(**kwargs) -> ComputeEngineRunContainerizedWorkloadSensor:
    return ComputeEngineRunContainerizedWorkloadSensor(
        task_id='a_step',
        instance_name='a-machine',
        container_image='an-image',
        **kwargs,
    )


def _email(sensor: ComputeEngineRunContainerizedWorkloadSensor) -> str:
    return sensor.declare_instance().service_accounts[0].email


def test_vm_service_account_is_the_attached_account() -> None:
    assert _email(_sensor(vm_service_account=VM_ACCOUNT)) == VM_ACCOUNT


def test_vm_service_account_wins_over_the_impersonation_chain() -> None:
    sensor = _sensor(vm_service_account=VM_ACCOUNT, impersonation_chain=['first@x.iam', CHAIN_ACCOUNT])
    assert _email(sensor) == VM_ACCOUNT
    assert sensor.impersonation_chain == ['first@x.iam', CHAIN_ACCOUNT]


def test_without_vm_service_account_the_chain_still_decides() -> None:
    assert _email(_sensor(impersonation_chain=['first@x.iam', CHAIN_ACCOUNT])) == CHAIN_ACCOUNT


def test_vm_service_account_defaults_to_none() -> None:
    assert _sensor().vm_service_account is None
