"""Airflow DAG to extract credible sets and a study index from eQTL Catalogue's finemapping results."""

from pathlib import Path

from orchestration.utils.staging import build_staging_dag

dag = build_staging_dag(Path(__file__).parent)


if __name__ == '__main__':
    dag.test()
