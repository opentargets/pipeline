"""Tests for the boot image of the vms that run pis and pts steps."""

from unittest.mock import MagicMock, patch

from orchestration.operators.gce import COS_IMAGE_FAMILY, ComputeEngineRunContainerizedWorkloadSensor


def _instance(work_disk_size_gb: int = 0):
    sensor = ComputeEngineRunContainerizedWorkloadSensor(
        task_id='a_step',
        instance_name='a-machine',
        container_image='an-image',
        work_disk_size_gb=work_disk_size_gb,
    )
    hook = MagicMock(_get_credentials_email='runner@a-project.iam.gserviceaccount.com')
    with patch.object(type(sensor), 'hook', property(lambda _: hook)):
        return sensor.declare_instance()


def test_the_boot_image_is_a_cos_lts_family() -> None:
    assert COS_IMAGE_FAMILY.startswith('projects/cos-cloud/global/images/family/cos-')
    assert COS_IMAGE_FAMILY.endswith('-lts')


def test_the_boot_disk_boots_from_the_family() -> None:
    instance = _instance()

    boot_disks = [d for d in instance.disks if d.boot]
    assert len(boot_disks) == 1
    assert boot_disks[0].initialize_params.source_image == COS_IMAGE_FAMILY


def test_the_work_disk_carries_no_image() -> None:
    instance = _instance(work_disk_size_gb=300)

    work_disks = [d for d in instance.disks if not d.boot]
    assert len(work_disks) == 1
    assert work_disks[0].device_name == 'work-disk'
    assert not work_disks[0].initialize_params.source_image
