"""Tests that the unified pipeline boots every cluster and vm from a pinned image."""

import re

import pytest

from orchestration.dags.config.unified_pipeline import UnifiedPipelineConfig
from orchestration.models.pts_step import pts_step_from_config
from orchestration.operators.gce import ComputeEngineRunContainerizedWorkloadSensor


@pytest.fixture(scope='module')
def config() -> UnifiedPipelineConfig:
    """The real pipeline config, so the tests read the shipped yaml."""
    return UnifiedPipelineConfig()


def test_every_cluster_pins_a_subminor_image(config: UnifiedPipelineConfig) -> None:
    """A bare `3.0` or `2.2` floats to whatever Google published last."""
    for name, cluster in config.clusters.config['clusters'].items():
        assert re.fullmatch(r'\d+\.\d+\.\d+-\w+', cluster['image_version']), f'{name}: {cluster["image_version"]}'


def test_every_vm_boots_the_pinned_cos_image(dag_bag, config: UnifiedPipelineConfig) -> None:
    """Checked on the DAG's tasks: a pin the operator never receives boots the family's newest image."""
    dag = dag_bag.dags['unified_pipeline']
    vm_steps = config.steps('pis_') + [s for s in config.steps('pts_') if pts_step_from_config(s, config).is_gce]
    for step_name in vm_steps:
        task = dag.get_task(f'{step_name}.run_{step_name}')
        assert task.source_image == config.cos_image, step_name
    assert len(vm_steps) > 1


def test_the_boot_disk_uses_the_source_image() -> None:
    sensor = ComputeEngineRunContainerizedWorkloadSensor(
        task_id='a_step',
        instance_name='a-machine',
        container_image='an-image',
        source_image='projects/cos-cloud/global/images/cos-an-image',
    )
    [boot_disk] = sensor.declare_instance().disks
    assert boot_disk.initialize_params.source_image == 'projects/cos-cloud/global/images/cos-an-image'
