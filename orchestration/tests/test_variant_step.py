"""The PySpark variant index runs on its own Dataproc cluster, after FoldX on a VM."""

from orchestration.dags.config.unified_pipeline import UnifiedPipelineConfig
from orchestration.models.pts_step import PTSDataprocStep, pts_step_from_config


def test_variant_runs_on_its_dataproc_cluster() -> None:
    config = UnifiedPipelineConfig()
    step = pts_step_from_config('pts_variant', config)
    assert isinstance(step, PTSDataprocStep)
    assert step.cluster_definition.cluster_type == 'pts_variant'


def test_foldx_runs_on_a_vm_before_the_variant_index() -> None:
    config = UnifiedPipelineConfig()
    assert pts_step_from_config('pts_foldx', config).is_gce
    assert config.step_definition('pts_foldx')['depends_on'] == ['pis_variant']
    assert {'pis_variant', 'pts_foldx', 'gentropy_variant_annotation'} <= set(
        config.step_definition('pts_variant')['depends_on']
    )
