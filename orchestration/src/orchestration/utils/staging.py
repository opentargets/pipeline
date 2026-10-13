"""Builder of the staging DAGs.

A staging DAG is defined by its `config.yaml` (see `orchestration.models.staging_dag`):

- a pps step uploads the pps config it runs with, and submits
  `pps -s <step> -c <step>.yaml` to its Dataproc cluster;
- a Batch step generates its batch index, and runs its Batch job mapped over it.
"""

from __future__ import annotations

from pathlib import Path

from airflow.sdk import DAG, BaseOperator
from airflow.sdk.definitions.mappedoperator import MappedOperator

from orchestration.dags.config.staging import DATAPROC_ENTRYPOINT, DATAPROC_INIT_SCRIPT, StagingDagConfig
from orchestration.operators.batch import BatchCollectOperator, BatchIndexOperator, BatchJobOperator
from orchestration.operators.dataproc import PTSJobBuilder, SubmitJobOperator
from orchestration.operators.gcs import UploadFileOperator, UploadStringOperator
from orchestration.utils.common import GCP_PROJECT_GENETICS, GCP_REGION, shared_dag_args, shared_dag_kwargs
from orchestration.utils.dataproc import create_cluster, delete_cluster
from orchestration.utils.labels import Labels
from orchestration.utils.utils import to_yaml

ASSETS_PATH = Path(__file__).parents[1] / 'assets'

Operator = BaseOperator | MappedOperator


def build_staging_dag(dag_dir: Path) -> DAG:
    """Build the staging DAG defined by `<dag_dir>/config.yaml`, named after `dag_dir`."""
    config = StagingDagConfig(dag_dir)

    with DAG(
        dag_id=config.dag_id,
        description=config.dag.description,
        default_args=shared_dag_args,
        **shared_dag_kwargs,
    ) as dag:
        # the first and last task of each step, to chain the steps between them
        starts: dict[str, Operator] = {}
        ends: dict[str, Operator] = {}

        for cluster_name in config.dag.infrastructure.clusters:
            steps = config.steps_on_cluster(cluster_name)
            if steps:
                runs = _cluster_steps(config, cluster_name, steps)
                starts |= runs
                ends |= runs

        for step_name, step in config.steps.items():
            if step.is_batch:
                starts[step_name], ends[step_name] = _batch_step(config, step_name)

        for step_name, step in config.steps.items():
            for dependency in step.depends_on:
                ends[dependency] >> starts[step_name]

    return dag


def _cluster_steps(config: StagingDagConfig, cluster_name: str, steps: list[str]) -> dict[str, BaseOperator]:
    """Create the tasks running pps steps on a Dataproc cluster, and the cluster around them."""
    upload_entrypoint = UploadFileOperator(
        task_id=f'upload_entrypoint_{cluster_name}',
        src_path=ASSETS_PATH / DATAPROC_ENTRYPOINT,
        dst_uri=config.bin_uri(DATAPROC_ENTRYPOINT),
    )
    upload_init_script = UploadFileOperator(
        task_id=f'upload_init_script_{cluster_name}',
        src_path=ASSETS_PATH / DATAPROC_INIT_SCRIPT,
        dst_uri=config.bin_uri(DATAPROC_INIT_SCRIPT),
    )
    create = create_cluster(cluster_name=cluster_name, cluster_config=config.cluster_config(cluster_name))
    delete = delete_cluster(cluster_name=cluster_name)
    [upload_entrypoint, upload_init_script] >> create

    runs: dict[str, BaseOperator] = {}
    for step_name in steps:
        upload_config = UploadStringOperator(
            task_id=f'upload_config_{step_name}',
            contents=to_yaml(config.step_config(step_name)),
            dst_uri=config.config_uri(step_name),
            overwrite=True,
        )
        run = SubmitJobOperator(
            task_id=step_name,
            project_id=GCP_PROJECT_GENETICS,
            region=GCP_REGION,
            cluster_name=cluster_name,
            step_name=step_name,
            py_spark_job=PTSJobBuilder(
                main_python_file_uri=config.bin_uri(DATAPROC_ENTRYPOINT),
                args=['-s', step_name, '-c', f'{step_name}.yaml'],
                config_uri=config.config_uri(step_name),
            ).build(),
            labels=Labels(project=GCP_PROJECT_GENETICS, extra={'step': step_name}),
            deferrable=True,
            execution_timeout=config.steps[step_name].execution_timeout,
        )
        [create, upload_config] >> run >> delete
        runs[step_name] = run
    return runs


def _batch_step(config: StagingDagConfig, step_name: str) -> tuple[Operator, Operator]:
    """Create the tasks of a Batch step, returning its first and last task."""
    step = config.steps[step_name]
    assert step.google_batch is not None
    assert step.google_batch_index_specs is not None

    index = BatchIndexOperator(
        task_id=f'{step_name}_index',
        batch_index_specs=step.google_batch_index_specs,
    )
    jobs = BatchJobOperator.partial(
        task_id=step_name,
        job_name=step_name.replace('_', '-'),
        batch_job_spec=step.google_batch,
        execution_timeout=step.execution_timeout,
    ).expand(batch_index_row=index.output)
    index >> jobs

    if step.google_batch.collect is None:
        return index, jobs
    collect = BatchCollectOperator(task_id=f'{step_name}_collect', collect_spec=step.google_batch.collect)
    jobs >> collect
    return index, collect
