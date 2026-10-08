"""Tests that the pts clusters, the Spark-NLP jar and pts' own pins agree on Spark 4."""

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


def test_pts_clusters_keep_ansi_mode_off(pts_clusters: dict[str, dict]) -> None:
    """Spark 4 defaults ANSI mode on, which turns today's null results into job failures."""
    for name, cluster in pts_clusters.items():
        assert cluster['properties'].get('spark:spark.sql.ansi.enabled') == 'false', name


def test_staged_spark_nlp_jar_is_the_scala_2_13_build(config: UnifiedPipelineConfig) -> None:
    """The 2.12 and 2.13 fat jars share a basename upstream; only the folder tells them apart."""
    nlp = {dst: src for dst, src in config.staged_jars.items() if 'spark-nlp' in dst}
    assert len(nlp) == 1, nlp
    [(dst, src)] = nlp.items()
    assert '/public/jars/scala-2.13/' in src, src
    assert Path(dst).name.startswith('spark-nlp-assembly_2.13-'), dst


def test_spark_nlp_jar_matches_the_pts_python_package(config: UnifiedPipelineConfig) -> None:
    """The jar the clusters load and the spark-nlp the init action installs must be one version.

    The clusters install pts without its lockfile, so the pyproject pin is what they get.
    """
    [dst] = [d for d in config.staged_jars if 'spark-nlp' in d]
    jar_version = Path(dst).stem.rsplit('-', 1)[1]
    assert _pts_pin('spark-nlp') == f'spark-nlp=={jar_version}'
