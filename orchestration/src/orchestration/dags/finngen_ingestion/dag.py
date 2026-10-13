"""Airflow DAG for transforming Finngen Susie finemapping to StudyLocus and generating finngen StudyIndex."""

from pathlib import Path

from orchestration.utils.staging import build_staging_dag

dag = build_staging_dag(Path(__file__).parent)


if __name__ == '__main__':
    dag.test()
