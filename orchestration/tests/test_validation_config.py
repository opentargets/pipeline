"""Study and credible-set validation run in PTS with PPP-specific inputs."""

import pytest

from orchestration.dags.config.unified_pipeline import UnifiedPipelineConfig

PPP_ONLY = ['sle_map', 'otar2077/microglia', 'eQTLGen/coeqtl', 'deCODE_proteomics']
BOTH = ['eqtl_catalogue/r8', 'ukb_ppp_eur']


@pytest.mark.parametrize('ppp', [False, True], ids=['platform', 'ppp'])
@pytest.mark.parametrize(('step', 'kind'), [('study', 'study'), ('credible_set', 'credible_set')])
def test_validation_inputs_follow_the_run(monkeypatch, ppp, step, kind):
    """PPP replaces the platform molQTL sources; both read released targets."""
    monkeypatch.setattr(UnifiedPipelineConfig, 'is_ppp', property(lambda self: ppp))
    config = UnifiedPipelineConfig()
    [task] = config.pts.config['steps'][step]
    sources = task['source'][kind]
    for name in PPP_ONLY:
        assert (f'input/molqtl/{kind}/{name}' in sources) is ppp
    for name in BOTH:
        assert f'input/molqtl/{kind}/{name}' in sources
    assert task['source']['target'] == 'output/target'
    if step == 'credible_set':
        assert task['source']['disease'] == 'output/disease'


def test_validation_steps_run_in_pts():
    config = UnifiedPipelineConfig()
    for step in ('study', 'credible_set', 'enhancer_to_gene'):
        assert f'pts_{step}' in config.steps('pts_')
        assert f'gentropy_{step}' not in config.steps()
    # pts_study already depends on pts_target and pts_disease, which credible set validation also reads.
    assert config.step_definition('pts_credible_set')['depends_on'] == ['pts_study']
