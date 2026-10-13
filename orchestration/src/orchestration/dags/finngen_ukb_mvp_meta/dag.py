"""Airflow DAG to ingest and harmonize FinnGen UKB meta-analysis data."""

from pathlib import Path

from orchestration.utils.staging import build_staging_dag

dag = build_staging_dag(Path(__file__).parent)


if __name__ == '__main__':
    dag.test()
