"""Tests for building a staging DAG from its config."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from airflow.sdk import DAG, BaseOperator
from airflow.task.trigger_rule import TriggerRule

from orchestration.dags.config.staging_pipeline import StagingDagConfig
from orchestration.models.batch import BatchIndexRow
from orchestration.models.batch.environment import EnvironmentRegistrySpec
from orchestration.utils import strhash
from orchestration.utils.common import GCP_PROJECT_GENETICS
from orchestration.utils.staging import build_staging_dag

if TYPE_CHECKING:
    from pathlib import Path

CONFIG = """
description: a staging dag
env: prod
environments:
  prod:
    input: {source: gs://source_inputs}
    output: gs://data_bucket/prod/20261026T153052
    scratchpad: {gentropy_version: 4.0.0-dev.1}
setup:
  - from: https://example.com/v4.0.0-dev.1/install_dependencies_on_cluster.sh
    to: '{{output}}/bin/install_dependencies_on_cluster.sh'
  - from: https://example.com/v4.0.0-dev.1/cli.py
    to: '{{output}}/bin/cli.py'
infrastructure:
  clusters:
    main:
      num_workers: 2
      metadata: {GENTROPY_REF: v4.0.0-dev.1}
      init_actions_uris: ['{{output}}/bin/install_dependencies_on_cluster.sh']
steps:
  first:
    kind: dataproc
    definition:
      cluster: main
      main: '{{output}}/bin/cli.py'
      args: |
        step=gentropy_first \\
          step.input_path={{input.source}}/studies \\
          step.output_path={{output}}/output/first
  second:
    depends_on: [first]
    kind: dataproc
    definition:
      cluster: main
      main: '{{output}}/bin/cli.py'
      args: |
        step=gentropy_second \\
          step.input_path={{output}}/output/first \\
          step.output_path={{output}}/output/second
"""
RUN = 'gs://data_bucket/prod/20261026T153052'
"""The run folder of the config."""
RUN_ID = 'manual__2026-10-26T15:30:52+00:00'
"""The id of a DAG run."""
FILTERS = {'strhash': strhash}
"""The Jinja filters of a staging DAG, for the names of the resources of a run."""


@pytest.fixture
def tasks(tmp_path: Path) -> dict[str, BaseOperator]:
    """Return the tasks of the DAG built from the config, by task id."""
    dag_dir = tmp_path / 'a_dag'
    dag_dir.mkdir()
    (dag_dir / 'config.yaml').write_text(CONFIG)
    config = StagingDagConfig(dag_dir)
    with DAG(dag_id=config.dag_id, schedule=None, user_defined_filters=FILTERS) as dag:
        build_staging_dag(config)
    return {task.task_id: task for task in dag.tasks if isinstance(task, BaseOperator)}


def rendered(task: Any) -> Any:
    """Render the templated fields of a task for the DAG run `RUN_ID`."""
    task.render_template_fields({'run_id': RUN_ID})
    return task


class TestBuildStagingDag:
    """The tasks of a staging DAG and the order they run in."""

    def test_tasks(self, tasks: dict[str, BaseOperator]) -> None:
        """The tasks of a run: the config and bin file uploads, the cluster, and the steps."""
        assert set(tasks) == {
            'upload_config',
            'upload_bin_install-dependencies-on-cluster-sh',
            'upload_bin_cli-py',
            'create_cluster_main',
            'first.run',
            'second.run',
            'delete_cluster_main',
        }

    def test_order(self, tasks: dict[str, BaseOperator]) -> None:
        """The cluster is created after the uploads, before the first step, and deleted after the last."""
        assert tasks['create_cluster_main'].upstream_task_ids == {
            'upload_config',
            'upload_bin_install-dependencies-on-cluster-sh',
            'upload_bin_cli-py',
        }
        assert tasks['first.run'].upstream_task_ids == {'create_cluster_main'}
        assert tasks['second.run'].upstream_task_ids == {'first.run'}
        assert tasks['delete_cluster_main'].upstream_task_ids == {'second.run'}

    def test_job_is_rendered_for_the_run(self, tasks: dict[str, BaseOperator]) -> None:
        """The job arguments are the gentropy CLI of the run and the paths of the run."""
        job = tasks['first.run'].py_spark_job  # ty: ignore[unresolved-attribute]
        assert job.main_python_file_uri == f'{RUN}/bin/cli.py'
        assert list(job.args)[:3] == [
            'step=gentropy_first',
            'step.input_path=gs://source_inputs/studies',
            f'step.output_path={RUN}/output/first',
        ]

    def test_cluster_is_rendered_for_the_run(self, tasks: dict[str, BaseOperator]) -> None:
        """The cluster init action is the init script of the run, with the gentropy tag in the metadata."""
        cluster_config = tasks['create_cluster_main'].cluster_config  # ty: ignore[unresolved-attribute]
        init_actions = [action['executable_file'] for action in cluster_config['initialization_actions']]
        assert init_actions == [f'{RUN}/bin/install_dependencies_on_cluster.sh']
        assert cluster_config['gce_cluster_config']['metadata']['GENTROPY_REF'] == 'v4.0.0-dev.1'

    def test_cluster_name_is_distinct_for_each_run(self, tasks: dict[str, BaseOperator]) -> None:
        """The cluster is created, used and deleted as `<cluster>-<hash of the DAG run id>`."""
        for task_id in ('create_cluster_main', 'first.run', 'second.run', 'delete_cluster_main'):
            assert rendered(tasks[task_id]).cluster_name == f'main-{strhash(RUN_ID)}', task_id

    def test_labels(self, tasks: dict[str, BaseOperator]) -> None:
        """The cluster and the jobs have the genetics labels and the DAG, with the cluster or the step."""
        cluster_labels = dict(tasks['create_cluster_main'].labels)  # ty: ignore[unresolved-attribute]
        job_labels = dict(tasks['first.run'].labels)  # ty: ignore[unresolved-attribute]
        assert cluster_labels['subteam'] == job_labels['subteam'] == 'genetics'
        assert cluster_labels['dag'] == job_labels['dag'] == 'a_dag'
        assert cluster_labels['cluster'] == 'main'
        assert job_labels['step'] == 'first'

    def test_uploads_are_rendered_for_the_run(self, tasks: dict[str, BaseOperator]) -> None:
        """The config and the bin files are uploaded to the folders of the run."""
        upload_config: Any = tasks['upload_config']
        assert str(upload_config.dst_uri) == f'{RUN}/config/a_dag.yaml'
        assert f'output: {RUN}\n' in upload_config.contents
        # the record is the text of the config, rendered, with its layout
        assert f'step.output_path={RUN}/output/first' in upload_config.contents
        assert 'args: |' in upload_config.contents
        upload_cli: Any = tasks['upload_bin_cli-py']
        assert str(upload_cli.dst_uri) == f'{RUN}/bin/cli.py'
        assert upload_cli.src_url == 'https://example.com/v4.0.0-dev.1/cli.py'


def test_runs_without_setup(tmp_path: Path) -> None:
    """Without setup, the config upload is the only task before the cluster."""
    config_yaml = (
        CONFIG.split('setup:', maxsplit=1)[0]
        + 'infrastructure:'
        + CONFIG
        .split('infrastructure:')[1]
        .replace("'{{output}}/bin/cli.py'", 'gs://code/cli.py')
        .replace("['{{output}}/bin/install_dependencies_on_cluster.sh']", '[]')
    )
    dag_dir = tmp_path / 'a_dag'
    dag_dir.mkdir()
    (dag_dir / 'config.yaml').write_text(config_yaml)
    config = StagingDagConfig(dag_dir)
    assert config.dag.setup == []
    with DAG(dag_id=config.dag_id, schedule=None, user_defined_filters=FILTERS) as dag:
        build_staging_dag(config)
    tasks = {task.task_id: task for task in dag.tasks}
    assert set(tasks) == {'upload_config', 'create_cluster_main', 'first.run', 'second.run', 'delete_cluster_main'}
    assert tasks['create_cluster_main'].upstream_task_ids == {'upload_config'}


VM_AND_BATCH_STEPS = """
  annotate:
    kind: vm
    definition:
      image: gentropy:4.0.0-dev.1
      machine_type: n2-highmem-16
      command: |
        gentropy step=annotate \\
          step.output_path={{output}}/output/annotation
  finemap:
    depends_on: [second]
    kind: batch
    definition:
      index:
        pointer: finemapping
        max_task_count: 100
        generator_specs:
          generator_options: {output_path: '{{output}}/output/susie'}
      job:
        job:
          task_group:
            parallelism: 10
            task_environments: {environments: []}
            task_config:
              instance_resource_spec: {cpu_milli: 4000, memory_mib: 25000, boot_disk_mib: 20000}
              runnable_spec:
                image_uri: gentropy:4.0.0-dev.1
                inline_commands: ['-c', 'gentropy step=susie step.output_path={{output}}/output/susie']
          allocation: {instance: {machine_type: n2-highmem-4}}
          logs: {}
        collect:
          source_prefix: '{{output}}/output/susie'
          destination_prefix: '{{output}}/output/credible_set'
"""
"""A VM step and a Batch step, appended to the steps of the config."""


@pytest.fixture
def vm_and_batch_tasks(tmp_path: Path) -> dict[str, Any]:
    """Return the tasks of the DAG built from the config with a VM step and a Batch step, by task id."""
    dag_dir = tmp_path / 'a_dag'
    dag_dir.mkdir()
    (dag_dir / 'config.yaml').write_text(CONFIG + VM_AND_BATCH_STEPS)
    config = StagingDagConfig(dag_dir)
    with DAG(dag_id=config.dag_id, schedule=None, user_defined_filters=FILTERS) as dag:
        build_staging_dag(config)
    return {task.task_id: task for task in dag.tasks}


class TestVmAndBatchSteps:
    """The task groups of the VM and Batch steps."""

    def test_tasks(self, vm_and_batch_tasks: dict[str, Any]) -> None:
        """A VM step is a VM run, its deletion and an end; a Batch step an index, its jobs and a collect."""
        assert {'annotate.run', 'annotate.delete_vm', 'annotate.done'} <= set(vm_and_batch_tasks)
        assert {'finemap.index', 'finemap.job', 'finemap.collect'} <= set(vm_and_batch_tasks)

    def test_order(self, vm_and_batch_tasks: dict[str, Any]) -> None:
        """A step without a cluster or dependencies follows the uploads, and a dependency precedes a group."""
        tasks = vm_and_batch_tasks
        assert tasks['annotate.run'].upstream_task_ids == {
            'upload_config',
            'upload_bin_install-dependencies-on-cluster-sh',
            'upload_bin_cli-py',
        }
        assert tasks['annotate.delete_vm'].upstream_task_ids == {'annotate.run'}
        assert tasks['annotate.delete_vm'].trigger_rule == TriggerRule.ALL_DONE
        assert tasks['annotate.done'].upstream_task_ids == {'annotate.run'}
        assert tasks['finemap.index'].upstream_task_ids == {'second.run'}
        assert tasks['finemap.job'].upstream_task_ids == {'finemap.index'}
        assert tasks['finemap.collect'].upstream_task_ids == {'finemap.job'}
        assert tasks['delete_cluster_main'].upstream_task_ids == {'second.run'}

    def test_labels(self, vm_and_batch_tasks: dict[str, Any]) -> None:
        """The VM and the Batch jobs have the genetics labels, the DAG and the step."""
        vm_labels = dict(vm_and_batch_tasks['annotate.run'].labels)
        job_labels = dict(vm_and_batch_tasks['finemap.job'].partial_kwargs['labels'])
        assert {key: vm_labels[key] for key in ('subteam', 'dag', 'step')} == {
            'subteam': 'genetics',
            'dag': 'a_dag',
            'step': 'annotate',
        }
        assert {key: job_labels[key] for key in ('subteam', 'dag', 'step')} == {
            'subteam': 'genetics',
            'dag': 'a_dag',
            'step': 'finemap',
        }

    def test_vm_is_in_the_genetics_project(self, vm_and_batch_tasks: dict[str, Any]) -> None:
        """The VM of a step is created and deleted in the genetics project, as the other resources."""
        assert vm_and_batch_tasks['annotate.run'].project == GCP_PROJECT_GENETICS
        assert vm_and_batch_tasks['annotate.delete_vm'].project_id == GCP_PROJECT_GENETICS

    def test_vm_is_rendered_for_the_run(self, vm_and_batch_tasks: dict[str, Any]) -> None:
        """The VM is named after the step and the run, and the command runs on the paths of the run."""
        run = rendered(vm_and_batch_tasks['annotate.run'])
        assert run.instance_name == f'staging-annotate-{strhash(RUN_ID)}'
        assert run.machine_type == 'n2-highmem-16'
        assert run.container_command == (f'gentropy step=annotate    step.output_path={RUN}/output/annotation')
        delete = rendered(vm_and_batch_tasks['annotate.delete_vm'])
        assert delete.resource_id == f'staging-annotate-{strhash(RUN_ID)}'

    def test_batch_job_is_rendered_for_the_run(self, vm_and_batch_tasks: dict[str, Any]) -> None:
        """The job of an index row, as Airflow builds a mapped task, runs on the paths of the run."""
        mapped = vm_and_batch_tasks['finemap.job']
        job = mapped.unmap({
            'batch_index_row': BatchIndexRow(idx=0, environments=EnvironmentRegistrySpec(environments=[]))
        })
        container = job.job.task_groups[0].task_spec.runnables[0].container
        assert list(container.commands) == ['-c', f'gentropy step=susie step.output_path={RUN}/output/susie']

    def test_batch_is_rendered_for_the_run(self, vm_and_batch_tasks: dict[str, Any]) -> None:
        """The index and the collect read and write the paths of the run."""
        index = vm_and_batch_tasks['finemap.index']
        assert index.generator_specs.generator_options == {'output_path': f'{RUN}/output/susie'}
        collect = vm_and_batch_tasks['finemap.collect']
        assert collect.collect_spec.source_prefix == f'{RUN}/output/susie'
        assert collect.collect_spec.destination_prefix == f'{RUN}/output/credible_set'


def test_builds_test_runs(tmp_path: Path) -> None:
    """A test run reads a static slice of the prod input, and writes to the test folder of the run."""
    dag_dir = tmp_path / 'a_dag'
    dag_dir.mkdir()
    test_env = (
        '  test:\n'
        '    input: {source: gs://data_bucket/fixtures/20261026T153000/source_inputs}\n'
        '    output: gs://data_bucket/test/20261026T153052\n'
        '    scratchpad: {gentropy_version: 4.0.0-dev.1}\n'
    )
    (dag_dir / 'config.yaml').write_text(
        CONFIG.replace('env: prod', 'env: test').replace('setup:', f'{test_env}setup:')
    )
    config = StagingDagConfig(dag_dir)
    with DAG(dag_id=config.dag_id, schedule=None, user_defined_filters=FILTERS):
        tasks = build_staging_dag(config)
    job = tasks['first'].children['first.run'].py_spark_job  # ty: ignore[unresolved-attribute]
    assert list(job.args)[1:3] == [
        'step.input_path=gs://data_bucket/fixtures/20261026T153000/source_inputs/studies',
        'step.output_path=gs://data_bucket/test/20261026T153052/output/first',
    ]
