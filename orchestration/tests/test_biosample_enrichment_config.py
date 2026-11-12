"""Biosample enrichment is a PPP-only step."""

import pytest

from orchestration.dags.config.unified_pipeline import UnifiedPipelineConfig


@pytest.mark.parametrize('ppp', [False, True], ids=['platform', 'ppp'])
def test_biosample_enrichment_runs_only_in_ppp(monkeypatch, ppp):
    monkeypatch.setattr(UnifiedPipelineConfig, 'is_ppp', property(lambda self: ppp))
    config = UnifiedPipelineConfig()
    assert ('pts_biosample_enrichment' in config.steps('pts_')) is ppp
    assert 'biosample_enrichment' in config.pts.config['steps']
