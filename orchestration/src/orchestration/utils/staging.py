"""Builder of the staging DAGs.

A staging DAG is defined by its `config.yaml` (see `orchestration.models.staging_dag`).
Each step is a task group named after the step, built the way the unified pipeline
builds its steps, and ending in an `end_<step>` task the steps depending on it wait for:

- a cluster step uploads its pps config and the files its cluster needs, creates the
  cluster (or reuses it, when an earlier step created it), and submits
  `pps -s <step> -c <step>.yaml` to it. A cluster is deleted once all its steps ended,
  and the jars it provides are staged before it is created;
- a VM step uploads its pps config, runs `pps -s <step> -c /config.yaml` from the pps
  image on a VM, and deletes the VM, whether the step succeeded or not;
- a Batch step generates its batch index, and runs its Batch job mapped over it.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.sdk import DAG, BaseOperator, TaskGroup, chain
from airflow.task.trigger_rule import TriggerRule

from orchestration.dags.config.staging import (
    DATAPROC_ENTRYPOINT,
    DATAPROC_INIT_SCRIPT,
    VM_CONFIG_PATH,
    StagingDagConfig,
)
from orchestration.operators.batch import BatchCollectOperator, BatchIndexOperator, BatchJobOperator
from orchestration.operators.dataproc import PTSJobBuilder, SubmitJobOperator
from orchestration.operators.gce import ComputeEngineRunContainerizedWorkloadSensor, DeleteInstanceOperator
from orchestration.operators.gcs import UploadFileOperator, UploadRemoteFileOperator, UploadStringOperator
from orchestration.utils.common import GCP_PROJECT_GENETICS, GCP_REGION, GCP_ZONE, shared_dag_args, shared_dag_kwargs
from orchestration.utils.dataproc import create_cluster, delete_cluster
from orchestration.utils.labels import Labels
from orchestration.utils.utils import clean_name, resource_name, strhash, to_yaml

ASSETS_PATH = Path(__file__).parents[1] / 'assets'


def build_staging_dag(dag_dir: Path) -> DAG:
    """Build the staging DAG defined by `<dag_dir>/config.yaml`, named after `dag_dir`."""
    config = StagingDagConfig(dag_dir)

    with DAG(
        dag_id=config.dag_id,
        description=config.dag.description,
        default_args=shared_dag_args,
        user_defined_filters={'strhash': strhash},
        **shared_dag_kwargs,
    ) as dag:
        # a registry of the first and last task of each step, to chain the steps between them
        steps: dict[str, dict[str, BaseOperator]] = {}
        clusters: dict[str, list[str]] = {}  # the steps that run on each cluster

        for step_name, step in config.steps.items():
            with TaskGroup(group_id=step_name):
                match step.kind:
                    case 'cluster':
                        assert step.cluster is not None
                        steps[step_name] = _cluster_step(config, step_name)
                        clusters.setdefault(step.cluster, []).append(step_name)
                    case 'vm':
                        steps[step_name] = _vm_step(config, step_name)
                    case 'batch':
                        steps[step_name] = _batch_step(config, step_name)

        # stage the jars a cluster provides before it is created, once per jar
        staged: dict[str, UploadRemoteFileOperator] = {}
        for cluster_name, steps_in_cluster in clusters.items():
            for src_url, dst_uri in config.staged_jars(cluster_name):
                if dst_uri not in staged:
                    staged[dst_uri] = UploadRemoteFileOperator(
                        task_id=f'stage_jar_{clean_name(dst_uri.rsplit("/", 1)[-1])}',
                        src_url=src_url,
                        dst_uri=dst_uri,
                        skip_if_exists=True,
                        # a failure would fail every step on the cluster, so a reset
                        # mid-transfer from maven central is retried
                        retries=3,
                        retry_delay=timedelta(minutes=2),
                    )
                for step_name in steps_in_cluster:
                    staged[dst_uri].set_downstream(steps[step_name]['create'])

        # delete a cluster after its steps have run
        for cluster_name, steps_in_cluster in clusters.items():
            x = delete_cluster(cluster_name=cluster_name)
            for step_name in steps_in_cluster:
                x.set_upstream(steps[step_name]['end'])

        for step_name, step in config.steps.items():
            for dependency in step.depends_on:
                steps[step_name]['start'].set_upstream(steps[dependency]['end'])

    return dag


def _labels(step_name: str) -> Labels:
    return Labels({'tool': 'pps', 'step': step_name}, project=GCP_PROJECT_GENETICS)


def _end(step_name: str) -> EmptyOperator:
    return EmptyOperator(task_id=f'end_{step_name}', trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS)


def _upload_config(config: StagingDagConfig, step_name: str) -> UploadStringOperator:
    return UploadStringOperator(
        task_id=f'upload_config_{step_name}',
        contents=to_yaml(config.step_config(step_name)),
        dst_uri=config.config_uri(step_name),
        overwrite=True,
    )


def _cluster_step(config: StagingDagConfig, step_name: str) -> dict[str, BaseOperator]:
    """Create the tasks running a pps step on its Dataproc cluster."""
    cluster_name = config.steps[step_name].cluster
    assert cluster_name is not None

    u = _upload_config(config, step_name)
    u2 = UploadFileOperator(
        task_id=f'upload_entrypoint_{step_name}',
        src_path=ASSETS_PATH / DATAPROC_ENTRYPOINT,
        dst_uri=config.bin_uri(DATAPROC_ENTRYPOINT),
    )
    u3 = UploadFileOperator(
        task_id=f'upload_init_script_{step_name}',
        src_path=ASSETS_PATH / DATAPROC_INIT_SCRIPT,
        dst_uri=config.bin_uri(DATAPROC_INIT_SCRIPT),
    )
    c = create_cluster(cluster_name=cluster_name, cluster_config=config.cluster_config(cluster_name))
    r = SubmitJobOperator(
        task_id=f'run_{step_name}',
        project_id=GCP_PROJECT_GENETICS,
        region=GCP_REGION,
        cluster_name=cluster_name,
        step_name=step_name,
        py_spark_job=PTSJobBuilder(
            main_python_file_uri=config.bin_uri(DATAPROC_ENTRYPOINT),
            args=['-s', step_name, '-c', f'{step_name}.yaml'],
            config_uri=config.config_uri(step_name),
        ).build(),
        labels=_labels(step_name),
        deferrable=True,
        execution_timeout=config.steps[step_name].execution_timeout,
    )
    e = _end(step_name)

    chain(u, u2, u3, c, r, e)
    return {'start': u, 'create': c, 'end': e}


def _vm_step(config: StagingDagConfig, step_name: str) -> dict[str, BaseOperator]:
    """Create the tasks running a pps step on a VM of its own."""
    vm = config.steps[step_name].vm
    assert vm is not None
    vm_name = resource_name(step_name)
    config_uri = config.config_uri(step_name)

    u = _upload_config(config, step_name)
    r = ComputeEngineRunContainerizedWorkloadSensor(
        task_id=f'run_{step_name}',
        project=GCP_PROJECT_GENETICS,
        zone=GCP_ZONE,
        instance_name=vm_name,
        labels=_labels(step_name),
        container_image=config.pps_image,
        container_args=['-s', step_name, '-c', VM_CONFIG_PATH],
        container_files={config_uri: VM_CONFIG_PATH},
        machine_type=vm.machine_type,
        work_disk_size_gb=vm.work_disk_size_gb,
        deferrable=True,
        execution_timeout=config.steps[step_name].execution_timeout,
    )
    t = DeleteInstanceOperator(
        task_id=f'delete_vm_{step_name}',
        project_id=GCP_PROJECT_GENETICS,
        zone=GCP_ZONE,
        resource_id=vm_name,
        trigger_rule=TriggerRule.NONE_SKIPPED,
    )
    e = _end(step_name)

    chain(u, r, t)
    chain(r, e)
    chain(t, e)
    return {'start': u, 'end': e}


def _batch_step(config: StagingDagConfig, step_name: str) -> dict[str, BaseOperator]:
    """Create the tasks running a Batch step."""
    step = config.steps[step_name]
    assert step.google_batch is not None
    assert step.google_batch_index_specs is not None

    with TaskGroup(group_id=f'{step_name}_batch_jobs') as batch_jobs:
        i = BatchIndexOperator(
            task_id=f'generate_manifest_{step_name}',
            batch_index_specs=step.google_batch_index_specs,
        )
        b = BatchJobOperator.partial(
            task_id=f'run_{step_name}',
            job_name=clean_name(step_name),
            batch_job_spec=step.google_batch,
            execution_timeout=step.execution_timeout,
        ).expand(batch_index_row=i.output)
        chain(i, b)
        if step.google_batch.collect is not None:
            bc = BatchCollectOperator(task_id=f'collect_{step_name}', collect_spec=step.google_batch.collect)
            chain(b, bc)
    e = _end(step_name)

    chain(batch_jobs, e)
    return {'start': i, 'end': e}
