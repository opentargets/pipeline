"""Airflow DAG that fine-maps GWAS Catalog sumstats with SuSiE on Google Batch, and QCs the credible sets."""

from pathlib import Path

from orchestration.utils.staging import build_staging_dag

dag = build_staging_dag(Path(__file__).parent)


if __name__ == '__main__':
    dag.test()
