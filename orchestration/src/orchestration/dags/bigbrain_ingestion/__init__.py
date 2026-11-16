"""Airflow DAG for the ingestion of the BigBrain brain eQTL and sQTL fine-mapping."""

from pathlib import Path

from airflow.sdk import DAG

from orchestration.dags.config.staging_pipeline import StagingDagConfig
from orchestration.utils import strhash
from orchestration.utils.common import shared_dag_args, shared_dag_kwargs
from orchestration.utils.staging import build_staging_dag

config = StagingDagConfig(Path(__file__).parent)

with DAG(
    dag_id=config.dag_id,
    description=config.dag.description,
    default_args=shared_dag_args,
    user_defined_filters={'strhash': strhash},
    max_active_runs=1,
    **shared_dag_kwargs,
) as dag:
    build_staging_dag(config)


if __name__ == '__main__':
    dag.test()
