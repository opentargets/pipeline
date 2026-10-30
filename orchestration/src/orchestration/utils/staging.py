"""Build the tasks of a staging DAG from its `StagingDagConfig`."""

from __future__ import annotations

from typing import TYPE_CHECKING

from airflow.task.trigger_rule import TriggerRule

from orchestration.operators.dataproc import CreateClusterOperator, DeleteClusterOperator
from orchestration.operators.gcs import UploadRemoteFileOperator, UploadStringOperator
from orchestration.utils.common import GCP_PROJECT_GENETICS, GCP_REGION
from orchestration.utils.labels import Labels
from orchestration.utils.utils import clean_name

if TYPE_CHECKING:
    from airflow.sdk import BaseOperator, TaskGroup

    from orchestration.dags.config.staging_pipeline import StagingDagConfig

RUN_RESOURCE_SUFFIX_LENGTH = len('-abcde')
"""Length of the suffix of `run_resource_name`, the hash of the DAG run id."""


def run_resource_name(name: str) -> str:
    """Return the name of a Google Cloud resource of a DAG run, such as a cluster or a VM.

    As `resource_name` of the unified pipeline, the name ends with a hash of the DAG run id,
    rendered by Airflow with the `strhash` filter of the DAG, so the resources of concurrent
    runs are distinct, and the name is the same when tasks are cleared.

    Examples:
        >>> run_resource_name('otg_tophit')
        'otg-tophit-{{ run_id | strhash }}'
    """
    return f'{clean_name(name)}-{{{{ run_id | strhash }}}}'


def staging_labels(dag_id: str, **extra: str) -> Labels:
    """Return the labels of a resource of a staging DAG: the genetics labels, the DAG, and `extra`.

    The operators add the `run` label, the id of the DAG run, when the resource is created.

    Examples:
        >>> labels = staging_labels('gwas_catalog_top_hits', step='pics')
        >>> {key: labels[key] for key in ('subteam', 'created_by', 'dag', 'step')}
        {'subteam': 'genetics', 'created_by': 'gentropy-pipelines', 'dag': 'gwas_catalog_top_hits', 'step': 'pics'}
    """
    return Labels(project=GCP_PROJECT_GENETICS, extra={'dag': dag_id, **extra})


def build_staging_dag(config: StagingDagConfig) -> dict[str, TaskGroup]:
    """Create the tasks of a staging DAG, inside the DAG context.

    The DAG renders the names of the resources of a run with the `strhash` filter, so it is
    created with `user_defined_filters={'strhash': strhash}`. The runs of one config write
    to the same run folder, so it is created with `max_active_runs=1`.

    The first tasks of a run upload the rendered config into the `config/` folder of the run,
    and the `setup` files, if any, into the `bin/` folder of the run. Each cluster is
    created before its first step and deleted after its last step, as `<cluster>-<hash>`, a
    name distinct for each DAG run (`run_resource_name`). A step without a cluster and
    without dependencies follows the uploads. The order of the steps follows `depends_on`,
    and the tasks of each step are the task group of `definition.build()`. Every resource has
    the labels of `staging_labels`: the labels of a step on the resources of the step, and the
    labels of a cluster on the cluster.

    Args:
        config (StagingDagConfig): The configuration of the DAG.

    Returns:
        dict[str, TaskGroup]: The task groups of the steps, by step name.
    """
    steps = config.dag.steps
    tasks = {
        name: step.definition.build(name, staging_labels(config.dag_id, step=name)) for name, step in steps.items()
    }

    setup: list[BaseOperator] = [
        UploadStringOperator(
            task_id='upload_config',
            contents=config.rendered_config(),
            dst_uri=config.config_uri,
            overwrite=True,
        ),
        *(
            UploadRemoteFileOperator(
                task_id=f'upload_bin_{clean_name(copy.file_name)}',
                src_url=copy.source,
                dst_uri=copy.to,
            )
            for copy in config.dag.setup
        ),
    ]

    for name, step in steps.items():
        for upstream in step.depends_on:
            tasks[upstream] >> tasks[name]
        if not step.depends_on and not step.definition.clusters:
            setup >> tasks[name]

    for cluster_name in config.dag.infrastructure.clusters:
        on_cluster = {name for name, step in steps.items() if cluster_name in step.definition.clusters}
        create = CreateClusterOperator(
            task_id=f'create_cluster_{cluster_name}',
            project_id=GCP_PROJECT_GENETICS,
            region=GCP_REGION,
            cluster_name=run_resource_name(cluster_name),
            cluster_config=config.cluster_config(cluster_name),
            labels=staging_labels(config.dag_id, cluster=cluster_name),
            deferrable=True,
        )
        delete = DeleteClusterOperator(
            task_id=f'delete_cluster_{cluster_name}',
            project_id=GCP_PROJECT_GENETICS,
            region=GCP_REGION,
            cluster_name=run_resource_name(cluster_name),
            trigger_rule=TriggerRule.ALL_SUCCESS,
            deferrable=True,
        )
        setup >> create
        for name in on_cluster:
            if not set(steps[name].depends_on) & on_cluster:
                create >> tasks[name]
            if not any(name in steps[other].depends_on for other in on_cluster):
                tasks[name] >> delete
    return tasks
