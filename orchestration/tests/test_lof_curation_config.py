"""PPP-only LoF curation stays connected across the Gentropy-to-PTS migration."""

import pytest

from orchestration.dags.config.unified_pipeline import UnifiedPipelineConfig


@pytest.mark.parametrize('ppp', [False, True], ids=['platform', 'ppp'])
def test_lof_ingestion_and_consumer_are_enabled_together(monkeypatch, ppp):
    """Standard runs omit LoF; PPP consumers read the generated release intermediate."""
    monkeypatch.setattr(UnifiedPipelineConfig, 'is_ppp', property(lambda self: ppp))
    config = UnifiedPipelineConfig()
    assert ('pis_variant_ppp' in config.steps('pis_')) is ppp
    assert ('pts_variant_ppp' in config.steps('pts_')) is ppp
    assert config.step_definition('pts_variant_ppp')['depends_on'] == ['pis_variant_ppp']
    variant = config.step_definition('gentropy_variant')
    assert 'pts_variant_ppp' not in variant['depends_on']
    assert 'pts_variant_ppp' in variant['depends_on_ppp']
    if ppp:
        annotations = config.gentropy.config['steps']['variant']['params']['step.variant_annotations_path']
        assert config.release_uri + '/input/variant/gnomad' in annotations
        assert config.release_uri + '/intermediate/variant/lof_curation' in annotations
