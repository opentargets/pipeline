"""Every gentropy cluster in the unified pipeline must run the release's gentropy version."""

from pathlib import Path

import yaml

from orchestration.dags.config import unified_pipeline
from orchestration.dags.config.unified_pipeline import UnifiedPipelineConfig


def test_every_gentropy_cluster_uses_gentropy_version() -> None:
    config = UnifiedPipelineConfig()
    up_yaml = Path(unified_pipeline.__file__).parent / 'unified_pipeline.yaml'
    version = yaml.safe_load(up_yaml.read_text())['gentropy_version']
    refs = {
        name: cluster['metadata']['GENTROPY_REF']
        for name, cluster in config.clusters.config['clusters'].items()
        if 'GENTROPY_REF' in cluster.get('metadata', {})
    }

    assert len(refs) > 1, refs
    assert refs == dict.fromkeys(refs, f'v{version}')
