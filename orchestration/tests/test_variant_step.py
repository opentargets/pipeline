"""The variant index runs on a VM, with FoldX in the same step."""

from orchestration.dags.config.unified_pipeline import UnifiedPipelineConfig
from orchestration.models.pts_step import pts_step_from_config


def test_variant_runs_on_a_vm() -> None:
    config = UnifiedPipelineConfig()
    assert pts_step_from_config('pts_variant', config).is_gce


def test_foldx_runs_inside_the_variant_step() -> None:
    config = UnifiedPipelineConfig()
    assert 'pts_foldx' not in config.steps('pts_')
    assert config.step_definition('pts_variant')['depends_on'] == ['pis_variant', 'gentropy_variant_annotation']
    foldx, index = config.pts.config['steps']['variant']
    assert foldx['transformer'] == 'foldx'
    assert index['requires'] == [foldx['name']]
