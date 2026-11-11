"""The variant index runs on a VM, with FoldX in the same step."""

import pytest

from orchestration.dags.config.unified_pipeline import UnifiedPipelineConfig
from orchestration.models.pts_step import pts_step_from_config


def test_variant_runs_on_a_vm() -> None:
    config = UnifiedPipelineConfig()
    assert pts_step_from_config('pts_variant', config).is_gce


def test_foldx_runs_inside_the_variant_step() -> None:
    config = UnifiedPipelineConfig()
    assert 'pts_foldx' not in config.steps('pts_')
    assert 'gentropy_variant_annotation' not in config.step_definition('pts_variant')['depends_on']
    foldx, index = config.pts.config['steps']['variant']
    assert foldx['transformer'] == 'foldx'
    assert index['requires'] == [foldx['name']]


@pytest.mark.parametrize('ppp', [False, True], ids=['platform', 'ppp'])
def test_the_variant_index_reads_the_validated_variants(monkeypatch, ppp) -> None:
    monkeypatch.setattr(UnifiedPipelineConfig, 'is_ppp', property(lambda self: ppp))
    _, index = UnifiedPipelineConfig().pts.config['steps']['variant']
    assert index['source']['credible_set_variant'] == 'intermediate/validated/credible_set_variant'
    assert 'credible_set' not in index['source']
