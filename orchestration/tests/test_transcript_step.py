"""pts_transcript reads only the human core dump that pis_target downloads, and the DAG says so."""

from pathlib import Path

import yaml

ROOT = Path(__file__).parents[2]
PTS_CONFIG = ROOT / 'pts/config.yaml'
DAG_CONFIG = ROOT / 'orchestration/src/orchestration/dags/config/unified_pipeline.yaml'

assert PTS_CONFIG.is_file(), PTS_CONFIG
assert DAG_CONFIG.is_file(), DAG_CONFIG

PTS = yaml.safe_load(PTS_CONFIG.read_text())['steps']
DAG = yaml.safe_load(DAG_CONFIG.read_text())['steps']

HUMAN_DUMP = 'input/ensembl/homo_sapiens'


def test_transcript_reads_only_the_human_core_dump() -> None:
    (task,) = PTS['transcript']
    assert task['source'] == {'ensembl': HUMAN_DUMP}


def test_pts_transcript_depends_on_pis_target_only() -> None:
    assert DAG['pts_transcript']['depends_on'] == ['pis_target']
