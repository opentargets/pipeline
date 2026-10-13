"""Airflow DAG for the preprocessing of GWAS Catalog's harmonised summary statistics and curated associations."""

from pathlib import Path

from orchestration.utils.staging import build_staging_dag

dag = build_staging_dag(Path(__file__).parent)


if __name__ == '__main__':
    dag.test()
