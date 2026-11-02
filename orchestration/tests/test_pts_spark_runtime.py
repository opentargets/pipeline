"""Tests that the pts clusters and pts' own pins agree on Spark 4."""

import re
import tomllib
from pathlib import Path

import pytest

from orchestration.dags.config.unified_pipeline import UnifiedPipelineConfig

# orchestration/tests/ -> the repository root, which holds the pts package
PTS_PYPROJECT = Path(__file__).resolve().parents[2] / 'pts' / 'pyproject.toml'


@pytest.fixture(scope='module')
def config() -> UnifiedPipelineConfig:
    """The real pipeline config, so the tests read the shipped yaml."""
    return UnifiedPipelineConfig()


@pytest.fixture(scope='module')
def pts_clusters(config: UnifiedPipelineConfig) -> dict[str, dict]:
    clusters = config.clusters.config['clusters']
    pts = {name: c for name, c in clusters.items() if name.startswith('pts')}
    assert pts, 'expected pts cluster definitions'
    return pts


def _pts_pin(package: str) -> str:
    assert PTS_PYPROJECT.is_file(), f'{PTS_PYPROJECT} not found'
    dependencies = tomllib.loads(PTS_PYPROJECT.read_text())['project']['dependencies']
    pins = [d for d in dependencies if re.match(rf'{package}\b', d)]
    assert len(pins) == 1, f'expected one {package} requirement in pts, found {pins}'
    return pins[0]


def test_pts_clusters_run_spark_4(pts_clusters: dict[str, dict]) -> None:
    """Every pts cluster is on the Spark 4 image; pts pins pyspark to its minor."""
    for name, cluster in pts_clusters.items():
        assert cluster['image_version'].startswith('3.0'), f'{name} is on {cluster["image_version"]}'
    assert _pts_pin('pyspark') == 'pyspark>=4.1,<4.2'


def test_pts_clusters_run_the_init_action_uploaded_into_the_run(
    config: UnifiedPipelineConfig,
    pts_clusters: dict[str, dict],
) -> None:
    """The clusters install pts with the repo's init action, not a hand-deployed copy."""
    assert config.pts_init_action_uri == f'{config.release_uri}/etc/bin/install_dependencies_on_cluster.sh'
    assert config.pts_init_action_source.is_file(), config.pts_init_action_source
    for name, cluster in pts_clusters.items():
        assert cluster['init_actions_uris'] == [config.pts_init_action_uri], name


def test_pts_clusters_keep_ansi_mode_off(pts_clusters: dict[str, dict]) -> None:
    """Spark 4 defaults ANSI mode on, which turns today's null results into job failures."""
    for name, cluster in pts_clusters.items():
        assert cluster['properties'].get('spark:spark.sql.ansi.enabled') == 'false', name


def test_pts_clusters_load_no_spark_nlp_jar(pts_clusters: dict[str, dict]) -> None:
    """No pts step uses spark-nlp, so no pts cluster loads its jar."""
    for name, cluster in pts_clusters.items():
        assert 'spark-nlp' not in cluster['properties'].get('spark:spark.jars', ''), name


def test_pts_clusters_install_no_extra(config: UnifiedPipelineConfig) -> None:
    """torch and the NER models are image-only; the clusters install pts' base dependencies alone."""
    assert '--extra' not in config.pts_init_action_source.read_text()
