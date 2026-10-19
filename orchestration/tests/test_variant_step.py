"""The variant step runs on a machine large enough for the in-memory variant index."""

from orchestration.dags.config.unified_pipeline import UnifiedPipelineConfig
from orchestration.models.pts_step import pts_step_from_config


def test_variant_runs_on_a_highmem_machine() -> None:
    config = UnifiedPipelineConfig()
    assert pts_step_from_config('pts_variant', config).machine_type == 'n2d-highmem-64'
