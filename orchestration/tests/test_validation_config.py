"""Study and credible-set validation run in PTS with PPP-specific inputs."""

import pytest

from orchestration.dags.config.unified_pipeline import UnifiedPipelineConfig

PPP_ONLY = ['sle_map', 'eqtl_catalogue/r8', 'otar2077/microglia', 'eQTLGen/coeqtl', 'deCODE_proteomics']
PLATFORM_ONLY = ['eqtl_catalogue/r7', 'otar2077/MAGE_OTAR', 'GTEx_v10']


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
    for name in PLATFORM_ONLY:
        assert (f'input/molqtl/{kind}/{name}' in sources) is not ppp
    assert task['source']['target'] == 'output/target'


def test_validation_steps_run_in_pts():
    config = UnifiedPipelineConfig()
    for step in ('study', 'credible_set', 'enhancer_to_gene'):
        assert f'pts_{step}' in config.steps('pts_')
        assert f'gentropy_{step}' not in config.steps()
    assert config.step_definition('pts_credible_set')['depends_on'] == ['pts_study', 'pts_target']
