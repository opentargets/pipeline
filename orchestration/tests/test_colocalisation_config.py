"""Colocalisation runs as a pts step on its own fixed cluster."""

from orchestration.dags.config.unified_pipeline import UnifiedPipelineConfig
from orchestration.models.pts_step import PTSDataprocStep


def test_colocalisation_runs_in_pts():
    config = UnifiedPipelineConfig()
    assert 'pts_colocalisation' in config.steps('pts_')
    assert 'gentropy_colocalisation' not in config.steps()
    assert config.step_definition('pts_colocalisation')['depends_on'] == ['pts_credible_set_annotation']
    assert 'pts_colocalisation' in config.step_definition('gentropy_l2g_feature_matrix')['depends_on']
    assert 'gentropy_colocalisation' not in config.step_definition('gentropy_l2g_feature_matrix')['depends_on']
    assert 'colocalisation' not in config.gentropy.config['steps']


def test_colocalisation_cluster_is_fixed_and_on_the_pts_image():
    config = UnifiedPipelineConfig()
    definition = PTSDataprocStep('pts_colocalisation', config).cluster_definition
    spec = definition.config
    pts_default = config.clusters.config['clusters']['pts']
    assert definition.cluster_type == 'pts_colocalisation'
    assert 'autoscaling_policy' not in spec
    assert 'secondary' not in ' '.join(spec)
    assert spec['num_workers'] == 32
    assert spec['image_version'] == pts_default['image_version']
    assert spec['properties']['spark:spark.sql.ansi.enabled'] == 'false'
    assert spec['init_actions_uris'] == pts_default['init_actions_uris']
    assert spec['metadata'] == pts_default['metadata']
    assert 'GENTROPY_REF' not in spec['metadata']
