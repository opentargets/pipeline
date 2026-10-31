"""The unified pipeline annotates variants inside the pts variant step, without Google Batch."""

import importlib
from pathlib import Path

import pytest
import yaml

from orchestration.dags.config.unified_pipeline import UnifiedPipelineConfig
from orchestration.models.pts_step import pts_step_from_config

REMOVED_STEPS = ('gentropy_variant_annotation', 'pts_variant_partition')

STANDALONE_BATCH_DAGS = (
    'orchestration.dags.gwas_catalog_harmonisation',
    'orchestration.dags.gwas_catalog_heritability_estimate',
    'orchestration.dags.gwas_catalog_sumstats_susie_finemapping',
    'orchestration.dags.ukb_ppp_eur_finemapping',
)


def test_unified_pipeline_has_no_vep_batch_step() -> None:
    config = UnifiedPipelineConfig()
    all_steps = config.steps('pis_') + config.steps('pts_') + config.steps('gentropy_')
    for removed in REMOVED_STEPS:
        assert removed not in all_steps
    config_file = Path(__file__).parents[1] / 'src/orchestration/dags/config/unified_pipeline.yaml'
    raw = yaml.safe_load(config_file.read_text())
    assert 'vep_version' not in raw


def test_variant_depends_on_the_step_sources_only() -> None:
    config = UnifiedPipelineConfig()
    assert config.step_definition('pts_variant')['depends_on'] == [
        'pis_variant',
        'pis_target',
        'pis_evidence_uniprot_variants',
        'pis_evidence_eva',
        'pts_pharmacogenetics',
        'pts_credible_set',
    ]


def test_variant_runs_on_a_64_core_vm() -> None:
    step = pts_step_from_config('pts_variant', UnifiedPipelineConfig())
    assert step.is_gce
    assert step.machine_type == 'n2d-standard-64'


@pytest.mark.parametrize('module', STANDALONE_BATCH_DAGS)
def test_standalone_dags_still_import_batch(module: str) -> None:
    importlib.import_module(module)


def test_variant_gets_a_500_gb_work_disk_and_other_steps_keep_the_default(dag_bag) -> None:
    config = UnifiedPipelineConfig()
    assert pts_step_from_config('pts_variant', config).disk_size == 500
    assert pts_step_from_config('pts_target', config).disk_size == config.pts_disk_size == 300
    dag = dag_bag.dags['unified_pipeline']
    assert dag.get_task('pts_variant.run_pts_variant').work_disk_size_gb == 500
    assert dag.get_task('pts_target.run_pts_target').work_disk_size_gb == 300


def test_every_gentropy_step_runs_on_a_cluster(dag_bag) -> None:
    config = UnifiedPipelineConfig()
    assert config.steps('gentropy_')
    for step_name in config.steps('gentropy_'):
        assert config.step_cluster_definition(step_name) is not None, step_name
    dag = dag_bag.dags['unified_pipeline']
    assert not [task.task_id for task in dag.tasks if type(task).__module__ == 'orchestration.operators.batch']
