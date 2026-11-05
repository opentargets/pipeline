"""Gentropy v4 reads the credible sets as two datasets, the sets and their variants."""

import pytest

from orchestration.dags.config.unified_pipeline import UnifiedPipelineConfig


@pytest.mark.parametrize('step', ['colocalisation', 'l2g_feature_matrix'])
def test_gentropy_reads_the_published_credible_sets_and_their_variants(step: str) -> None:
    params = UnifiedPipelineConfig().gentropy.config['steps'][step]['params']
    release = UnifiedPipelineConfig().release_uri
    assert params['step.credible_set_path'] == f'{release}/output/credible_set'
    assert params['step.credible_set_variant_path'] == f'{release}/output/credible_set_variant'
