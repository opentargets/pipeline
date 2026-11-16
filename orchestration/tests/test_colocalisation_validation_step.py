"""Colocalisation, written by gentropy, is validated by a pts step of its own before L2G reads it."""

from pathlib import Path

import yaml

ROOT = Path(__file__).parents[2]
PTS_CONFIG = ROOT / 'pts/config.yaml'
DAG_CONFIG = ROOT / 'orchestration/src/orchestration/dags/config/unified_pipeline.yaml'

PTS = yaml.safe_load(PTS_CONFIG.read_text())['steps']
DAG = yaml.safe_load(DAG_CONFIG.read_text())['steps']


def test_the_pts_step_reads_colocalisation_and_writes_nothing() -> None:
    (task,) = PTS['colocalisation_validation']
    assert task['transformer'] == 'schema_validation'
    assert task['source'] == {'colocalisation': 'output/colocalisation'}
    assert task['destination'] == {}


def test_validation_runs_after_gentropy_colocalisation() -> None:
    assert DAG['pts_colocalisation_validation']['depends_on'] == ['gentropy_colocalisation']


def test_l2g_reads_colocalisation_only_once_it_is_validated() -> None:
    depends_on = DAG['gentropy_l2g_feature_matrix']['depends_on']
    assert 'pts_colocalisation_validation' in depends_on
    assert 'gentropy_colocalisation' not in depends_on
