"""Study and credible-set validation run in PTS with PPP-specific inputs."""

from typing import Any

import pytest

from orchestration.dags.config.unified_pipeline import UnifiedPipelineConfig

PPP_ONLY = ['sle_map', 'otar2077/microglia', 'eQTLGen/coeqtl', 'deCODE_proteomics']
BOTH = ['eqtl_catalogue', 'ukb_ppp_eur']


@pytest.mark.parametrize(
    ('ppp', 'step', 'kind'),
    [(False, 'study', 'study'), (True, 'study', 'study'), (True, 'credible_set', 'credible_set')],
    ids=['study-platform', 'study-ppp', 'credible_set-ppp'],
)
def test_validation_inputs_follow_the_run(monkeypatch, ppp, step, kind):
    """PPP replaces the platform molQTL sources; both read released targets."""
    monkeypatch.setattr(UnifiedPipelineConfig, 'is_ppp', property(lambda self: ppp))
    config = UnifiedPipelineConfig()
    [task] = config.pts.config['steps'][step]
    sources = task['source'][kind]
    for name in PPP_ONLY:
        assert (f'input/molqtl/{name}/{kind}' in sources) is ppp
    for name in BOTH:
        assert f'input/molqtl/{name}/{kind}' in sources
    assert task['source']['target'] == 'output/target'
    if step == 'credible_set':
        assert task['source']['disease'] == 'output/disease'


PIS_PPP_STEPS = ('gwas_ppp', 'molqtl_ppp')
"""The PIS steps that stage inputs only for PPP runs."""


def _destinations(tasks: list[dict[str, Any]]) -> set[str]:
    """The destinations of PIS tasks, with `foreach` tasks expanded for every `${each}` value."""
    destinations = set()
    for task in tasks:
        if 'foreach' in task:
            for each in task['foreach']:
                destinations |= {path.replace('${each}', each) for path in _destinations(task['do'])}
        elif 'destination' in task:
            destinations.add(task['destination'])
    return destinations


def _expanded(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The tasks of a step, with every explode replaced by the tasks it generates for each value."""
    expanded = []
    for task in tasks:
        if 'foreach' in task:
            for each in task['foreach']:
                expanded += _expanded([_substitute(spec, each) for spec in task['do']])
        else:
            expanded.append(task)
    return expanded


def _substitute(value: Any, each: str) -> Any:
    if isinstance(value, str):
        return value.replace('${each}', each)
    if isinstance(value, list):
        return [_substitute(item, each) for item in value]
    if isinstance(value, dict):
        return {key: _substitute(item, each) for key, item in value.items()}
    return value


def _source_paths(source: Any) -> list[str]:
    if isinstance(source, str):
        return [source]
    if isinstance(source, list):
        return [path for item in source for path in _source_paths(item)]
    if isinstance(source, dict):
        return [path for value in source.values() for path in _source_paths(value)]
    return []


def test_platform_pts_reads_no_ppp_input(monkeypatch):
    """No platform PTS task reads a dataset that only the PPP PIS steps stage."""
    monkeypatch.setattr(UnifiedPipelineConfig, 'is_ppp', property(lambda self: False))
    config = UnifiedPipelineConfig()
    pis_steps = config.pis.config['steps']
    ppp_only = set().union(*(_destinations(pis_steps[step]) for step in PIS_PPP_STEPS)) - set().union(
        *(_destinations(tasks) for step, tasks in pis_steps.items() if step not in PIS_PPP_STEPS)
    )
    assert ppp_only, 'expected the PPP PIS steps to stage some inputs'
    read = {
        (step, task['name'], path)
        for step, tasks in config.pts.config['steps'].items()
        for task in _expanded(tasks)
        for path in _source_paths(task.get('source'))
    }
    assert {
        (step, name, path)
        for step, name, path in read
        if any(path == ppp or path.startswith(f'{ppp}/') for ppp in ppp_only)
    } == set()


def test_validation_steps_run_in_pts():
    config = UnifiedPipelineConfig()
    for step in ('study', 'credible_set', 'enhancer_to_gene'):
        assert f'pts_{step}' in config.steps('pts_')
        assert f'gentropy_{step}' not in config.steps()
    # pts_study already depends on pts_target and pts_disease, which credible set validation also reads.
    assert config.step_definition('pts_credible_set')['depends_on'] == ['pts_study']
