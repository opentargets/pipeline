"""The variant index runs on a VM, after FoldX."""

from orchestration.dags.config.unified_pipeline import UnifiedPipelineConfig
from orchestration.models.pts_step import pts_step_from_config


def test_variant_and_foldx_run_on_vms() -> None:
    config = UnifiedPipelineConfig()
    assert pts_step_from_config('pts_variant', config).is_gce
    assert pts_step_from_config('pts_foldx', config).is_gce


def test_foldx_runs_before_the_variant_index() -> None:
    config = UnifiedPipelineConfig()
    assert config.step_definition('pts_foldx')['depends_on'] == ['pis_variant']
    # pis_variant is reached through pts_foldx, so the index does not list it again.
    assert config.step_definition('pts_variant')['depends_on'] == ['pts_foldx', 'gentropy_variant_annotation']
