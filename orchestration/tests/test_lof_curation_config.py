"""PPP-only LoF curation stays connected across the Gentropy-to-PTS migration."""

from pathlib import Path

import pytest
from airflow.models import DagBag

from orchestration.dags.config.unified_pipeline import UnifiedPipelineConfig


@pytest.mark.parametrize('ppp', [False, True], ids=['platform', 'ppp'])
def test_lof_ingestion_and_consumer_are_enabled_together(monkeypatch, ppp):
    """Standard runs omit LoF; PPP consumers read the generated release intermediate."""
    monkeypatch.setattr(UnifiedPipelineConfig, 'is_ppp', property(lambda self: ppp))
    config = UnifiedPipelineConfig()
    assert ('pis_variant_ppp' in config.steps('pis_')) is ppp
    assert ('pts_variant_ppp' in config.steps('pts_')) is ppp
    assert config.step_definition('pts_variant_ppp')['depends_on'] == ['pis_variant_ppp']
    variant = config.step_definition('pts_variant')
    assert 'pts_variant_ppp' not in variant['depends_on']
    assert 'pts_variant_ppp' in variant['depends_on_ppp']
    tasks = {task['transformer']: task for task in config.pts.config['steps']['variant']}
    # Overrides replace whole lists, so PPP must keep FoldX alongside the index.
    assert tasks['foldx']['destination'] == 'intermediate/variant/foldx'
    assert tasks['variant']['requires'] == [tasks['foldx']['name']]
    annotations = tasks['variant']['source']['variant_annotations']
    assert 'input/variant/gnomad' in annotations
    assert ('intermediate/variant/lof_curation' in annotations) is ppp


def test_ppp_dag_connects_raw_ingestion_to_lof_and_variant(monkeypatch):
    """The PPP DAG loads and gates Variant on the completed LoF transformer."""
    monkeypatch.setattr(UnifiedPipelineConfig, 'is_ppp', property(lambda self: True))
    folder = Path(__file__).parents[1] / 'src/orchestration/dags'
    bag = DagBag(dag_folder=str(folder))
    assert not bag.import_errors
    dag = bag.dags['unified_pipeline']
    lof = dag.get_task('pts_variant_ppp.diff_pts_variant_ppp')
    variant = dag.get_task('pts_variant.diff_pts_variant')
    assert 'pis_variant_ppp.end_pis_variant_ppp' in lof.upstream_task_ids
    assert 'pts_variant_ppp.end_pts_variant_ppp' in variant.upstream_task_ids
