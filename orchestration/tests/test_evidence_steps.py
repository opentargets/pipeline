"""The pts evidence steps in config match the DAG: names, ppp-only flags, edges."""

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parents[2]
PTS_CONFIG = ROOT / 'pts/config.yaml'
DAG_CONFIG = ROOT / 'orchestration/src/orchestration/dags/config/unified_pipeline.yaml'

assert PTS_CONFIG.is_file(), PTS_CONFIG
assert DAG_CONFIG.is_file(), DAG_CONFIG

PTS = yaml.safe_load(PTS_CONFIG.read_text())['steps']
DAG = yaml.safe_load(DAG_CONFIG.read_text())['steps']
EVIDENCE_STEPS = sorted(name for name in DAG if name.startswith('pts_evidence_'))
LITERATURE_PREFIXES = ('input/literature', 'intermediate/literature')
DIRECT_ONLY_STEPS = [*EVIDENCE_STEPS, 'pts_l2g']


def dependencies(name: str) -> set[str]:
    step = DAG.get(name) or {}
    return set(step.get('depends_on') or []) | set(step.get('depends_on_ppp') or [])


def ancestors(name: str) -> set[str]:
    seen: set[str] = set()
    todo = list(dependencies(name))
    while todo:
        dep = todo.pop()
        if dep not in seen:
            seen.add(dep)
            todo.extend(dependencies(dep))
    return seen


def reads_literature(node: object) -> bool:
    if isinstance(node, str):
        return node.startswith(LITERATURE_PREFIXES)
    if isinstance(node, dict):
        return any(reads_literature(v) for v in node.values())
    if isinstance(node, list):
        return any(reads_literature(v) for v in node)
    return False


def test_pts_evidence_steps_match_the_dag() -> None:
    config_steps = {f'pts_{name}' for name in PTS if name.startswith('evidence_')}
    assert config_steps == set(EVIDENCE_STEPS)


def test_ppp_only_evidence_steps_stay_ppp_only() -> None:
    for name in ['pts_evidence_encore', 'pts_evidence_validation_lab', 'pts_evidence_ot_crispr']:
        assert DAG[name].get('ppp_only') is True, name


@pytest.mark.parametrize('name', DIRECT_ONLY_STEPS)
def test_no_dependency_is_already_implied(name: str) -> None:
    direct = dependencies(name)
    redundant = sorted(d for d in direct if any(d in ancestors(other) for other in direct if other != d))
    assert not redundant, f'{name} lists {redundant}, already reachable through another dependency'


@pytest.mark.parametrize('name', EVIDENCE_STEPS)
def test_literature_edge_iff_the_step_reads_literature(name: str) -> None:
    tasks = PTS[name.removeprefix('pts_')]
    assert reads_literature(tasks) == ('pis_literature' in ancestors(name)), name


ONTOMA_TIER1_SOURCES = ['gene2phenotype', 'orphanet', 'impc', 'panel_app']


@pytest.mark.parametrize('source', ONTOMA_TIER1_SOURCES)
def test_ontoma_tier1_sources_have_no_postprocess_step(source: str) -> None:
    assert f'pts_evidence_postprocess_{source}' not in DAG
    assert f'evidence_postprocess_{source}' not in PTS


@pytest.mark.parametrize('source', ONTOMA_TIER1_SOURCES)
def test_association_depends_on_ontoma_tier1_sources(source: str) -> None:
    assert f'pts_evidence_{source}' in dependencies('pts_association')


@pytest.mark.parametrize('step', [*(f'evidence_{s}' for s in ONTOMA_TIER1_SOURCES), 'pharmacogenetics'])
def test_ontoma_tier1_steps_have_no_pyspark_task(step: str) -> None:
    assert PTS[step], step
    assert all('pyspark' not in task for task in PTS[step]), step
