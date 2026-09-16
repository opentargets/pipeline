"""Airflow DAG that uses Google Cloud Batch to run the SuSiE Finemapper step for UKB PPP."""

from pathlib import Path

from airflow.sdk import DAG

from orchestration.models.batch import BatchIndexOperatorSpec, BatchJobOperatorSpec
from orchestration.operators.batch import BatchIndexOperator, BatchJobOperator
from orchestration.types import Environment, EnvironmentSpec
from orchestration.utils import chain_dependencies, find_environment_vars, read_yaml_config
from orchestration.utils.common import shared_dag_args, shared_dag_kwargs
from orchestration.utils.dataproc import generate_dataproc_task_chain, submit_gentropy_step

SOURCE_CONFIG_FILE_PATH = Path(__file__).parent / 'config' / 'ukb_ppp_eur_finemapping.yaml'
config = read_yaml_config(SOURCE_CONFIG_FILE_PATH)
env_spec: list[EnvironmentSpec] = config['environment_specs']
env: Environment = config['env']
sentinels = find_environment_vars(env_spec, env)
config = read_yaml_config(SOURCE_CONFIG_FILE_PATH, sentinels)

with DAG(
    dag_id=Path(__file__).stem,
    description='Open Targets Genetics — Susie Finemap UKB PPP (EUR)',
    default_args=shared_dag_args,
    **shared_dag_kwargs,
) as dag:
    tasks = {}
    for step in config['nodes']:
        match step['id']:
            case 'generate_finemapping_index':
                batch_index = BatchIndexOperator(
                    task_id=step['id'],
                    batch_index_specs=BatchIndexOperatorSpec(**step['google_batch_index_specs']),
                )
                task = batch_index
            case 'finemapping_batch_job':
                finemapping_job = BatchJobOperator.partial(
                    task_id=step['id'],
                    job_name='susie-finemapping',
                    batch_job_spec=BatchJobOperatorSpec(**step['google_batch']),
                ).expand(batch_index_row=batch_index.output)
                task = finemapping_job

            case _:
                task = submit_gentropy_step(
                    cluster_name=config['dataproc']['cluster_name'],
                    step_name=step['id'],
                    params=step['params'],
                )
                generate_dataproc_task_chain(tasks=[task], **config['dataproc'])

        tasks[step['id']] = task
    chain_dependencies(nodes=config['nodes'], tasks_or_task_groups=tasks)


if __name__ == '__main__':
    dag.test()
