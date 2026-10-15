"""PPP-only LoF curation stays connected across the Gentropy-to-PTS migration."""

import pytest

from orchestration.dags.config.unified_pipeline import UnifiedPipelineConfig


@pytest.mark.parametrize('ppp', [False, True], ids=['platform', 'ppp'])
def test_lof_ingestion_and_consumer_are_enabled_together(monkeypatch, ppp):
    """Standard runs omit LoF; PPP consumers read the generated release intermediate."""
    monkeypatch.setattr(UnifiedPipelineConfig, 'is_ppp', property(lambda self: ppp))
    config = UnifiedPipelineConfig()
    assert ('pis_lof_curation' in config.steps('pis_')) is ppp
    assert ('pts_lof_curation' in config.steps('pts_')) is ppp
    assert config.step_definition('pts_lof_curation')['depends_on'] == ['pis_lof_curation']
    variant_name = 'pts_variant' if 'pts_variant' in config.steps() else 'gentropy_variant'
    variant = config.step_definition(variant_name)
    assert 'pts_lof_curation' not in variant['depends_on']
    assert 'pts_lof_curation' in variant['depends_on_ppp']
    # Before the last PR, Gentropy consumes annotations; afterwards PTS does.
    if 'variant' in config.pts.config['steps']:
        annotations = config.pts.config['steps']['variant'][0]['source']['variant_annotations']
        assert ('intermediate/variant/lof_curation' in annotations) is ppp
    elif ppp:
        annotations = config.gentropy.config['steps']['variant']['params']['step.variant_annotations_path']
        assert config.release_uri + '/input/variant/gnomad' in annotations
        assert config.release_uri + '/intermediate/variant/lof_curation' in annotations


def test_ppp_dag_connects_raw_ingestion_to_lof_and_variant(monkeypatch):
    """The final PPP DAG loads and gates Variant on the completed LoF transformer."""
    from pathlib import Path

    from airflow.models import DagBag

    monkeypatch.setattr(UnifiedPipelineConfig, 'is_ppp', property(lambda self: True))
    folder = Path(__file__).parents[1] / 'src/orchestration/dags'
    bag = DagBag(dag_folder=str(folder))
    assert not bag.import_errors
    dag = bag.dags['unified_pipeline']
    lof = dag.get_task('pts_lof_curation.diff_pts_lof_curation')
    variant = dag.get_task('pts_variant.diff_pts_variant')
    assert 'pis_lof_curation.end_pis_lof_curation' in lof.upstream_task_ids
    assert 'pts_lof_curation.end_pts_lof_curation' in variant.upstream_task_ids
