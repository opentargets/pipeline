"""Airflow DAG that uses Google Cloud Batch to run the SuSiE Finemapper step for UKB PPP."""

from pathlib import Path

from orchestration.utils.staging import build_staging_dag

dag = build_staging_dag(Path(__file__).parent)


if __name__ == '__main__':
    dag.test()
