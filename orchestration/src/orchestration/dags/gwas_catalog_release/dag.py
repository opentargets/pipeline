"""Airflow DAG syncing the latest GWAS Catalog release and preparing its study curation."""

from pathlib import Path

from orchestration.utils.staging import build_staging_dag

dag = build_staging_dag(Path(__file__).parent)


if __name__ == '__main__':
    dag.test()
