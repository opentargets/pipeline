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
    assert ('pis_reference_grch38' in config.steps('pis_')) is ppp
    assert ('pts_reference_grch38' in config.steps('pts_')) is ppp
    assert config.step_definition('pts_reference_grch38')['depends_on'] == ['pis_reference_grch38']
    assert config.step_definition('pts_lof_curation')['depends_on'] == ['pis_lof_curation', 'pts_reference_grch38']
    download = config.pis.config['steps']['reference_grch38'][0]
    reference = config.pts.config['steps']['reference_grch38'][0]
    curation = config.pts.config['steps']['lof_curation'][0]
    assert download['source'] == 'gs://gcp-public-data--broad-references/hg38/v0/Homo_sapiens_assembly38.fasta'
    assert download['destination'] == 'input/reference/GRCh38/Homo_sapiens_assembly38.fasta'
    assert reference['source'] == download['destination']
    assert reference['destination'] == 'intermediate/reference/GRCh38'
    assert reference['transformer'] == 'reference'
    assert curation['source'] == {
        'curation': 'input/lof_curation/curation.csv',
        'reference': reference['destination'],
    }
    variant = config.step_definition('gentropy_variant')
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
