"""pts_disease_location reads MONDO plus the disease and biosample outputs, and the DAG says so."""

from pathlib import Path

import yaml

ROOT = Path(__file__).parents[2]
PTS_CONFIG = ROOT / 'pts/config.yaml'
DAG_CONFIG = ROOT / 'orchestration/src/orchestration/dags/config/unified_pipeline.yaml'

assert PTS_CONFIG.is_file(), PTS_CONFIG
assert DAG_CONFIG.is_file(), DAG_CONFIG

PTS = yaml.safe_load(PTS_CONFIG.read_text())['steps']
DAG = yaml.safe_load(DAG_CONFIG.read_text())['steps']


def test_disease_location_reads_mondo_disease_and_biosample() -> None:
    (task,) = PTS['disease_location']
    assert task['transformer'] == 'disease_location'
    assert task['source'] == {
        'mondo': 'input/disease/mondo.json',
        'disease': 'output/disease',
        'biosample': 'output/biosample',
    }


def test_pts_disease_location_depends_on_disease_and_biosample() -> None:
    assert sorted(DAG['pts_disease_location']['depends_on']) == ['pts_biosample', 'pts_disease']
