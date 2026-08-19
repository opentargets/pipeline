"""Airflow DAG to extract credible sets and a study index from eQTL Catalogue's finemapping results."""

from __future__ import annotations

from pathlib import Path

from airflow.sdk import DAG

from orchestration.utils import chain_dependencies, read_yaml_config
from orchestration.utils.common import shared_dag_args, shared_dag_kwargs
from orchestration.utils.dataproc import generate_dataproc_task_chain, submit_gentropy_step

CONFIG_PATH = Path(__file__).parent / 'config' / 'eqtl_catalogue_ingestion.yaml'
config = read_yaml_config(CONFIG_PATH)

with DAG(
    dag_id=Path(__file__).stem,
    description='Open Targets Genetics — eQTL credible set ingestion',
    default_args=shared_dag_args,
    **shared_dag_kwargs,
) as dag:
    tasks = {}
    for step in config['nodes']:
        task = submit_gentropy_step(
            cluster_name=config['dataproc']['cluster_name'],
            step_name=step['id'],
            params=step['params'],
        )
        tasks[step['id']] = task
    chain_dependencies(nodes=config['nodes'], tasks_or_task_groups=tasks)
    generate_dataproc_task_chain(tasks=list(tasks.values()), **config['dataproc'])

    if __name__ == '__main__':
        dag.test()
