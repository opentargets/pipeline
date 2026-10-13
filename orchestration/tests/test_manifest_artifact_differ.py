"""Tests for ManifestArtifactDiffer, in particular list-shaped destination handling.

Steps with more than one named destination (e.g. `ontoma`) get recorded in the
manifest as a single artifact whose `destination` is a list of URIs rather than
one string. is_diff() must handle both shapes without raising.
"""

import json
from unittest.mock import MagicMock

from orchestration.operators.differs.manifest_artifact_differ import ManifestArtifactDiffer


def _config_for(manifest: dict, tmp_path) -> MagicMock:
    manifest_path = tmp_path / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest))
    config = MagicMock()
    config.manifest_uri.return_value = str(manifest_path)
    return config


def _client_where_blobs_exist() -> MagicMock:
    client = MagicMock()
    blob = MagicMock()
    blob.exists.return_value = True
    client.bucket.return_value.blob.return_value = blob
    return client


def test_string_destination_all_present(tmp_path):
    """Existing single-string destination behaviour is unchanged."""
    manifest = {
        'steps': {
            'pts_example': {
                'artifacts': [{'destination': 'gs://bucket/output/example'}],
            }
        }
    }
    differ = ManifestArtifactDiffer()
    result = differ.is_diff(
        step_name='pts_example',
        config=_config_for(manifest, tmp_path),
        client=_client_where_blobs_exist(),
    )
    assert result is False


def test_list_destination_all_present(tmp_path):
    """A step with multiple named destinations records one artifact with a list destination.

    This used to raise AttributeError.
    """
    manifest = {
        'steps': {
            'pts_ontoma': {
                'artifacts': [
                    {
                        'destination': [
                            'gs://bucket/intermediate/ontoma/disease_id_lookup_table.parquet',
                            'gs://bucket/intermediate/ontoma/disease_label_lookup_table.parquet',
                        ]
                    }
                ],
            }
        }
    }
    differ = ManifestArtifactDiffer()
    result = differ.is_diff(
        step_name='pts_ontoma',
        config=_config_for(manifest, tmp_path),
        client=_client_where_blobs_exist(),
    )
    assert result is False


def test_list_destination_missing_artifact_triggers_diff(tmp_path):
    """If any URI in a list destination is missing, the step is considered diffed."""
    manifest = {
        'steps': {
            'pts_ontoma': {
                'artifacts': [
                    {
                        'destination': [
                            'gs://bucket/intermediate/ontoma/disease_id_lookup_table.parquet',
                            'gs://bucket/intermediate/ontoma/disease_label_lookup_table.parquet',
                        ]
                    }
                ],
            }
        }
    }
    differ = ManifestArtifactDiffer()
    client = MagicMock()
    missing_blob = MagicMock()
    missing_blob.exists.return_value = False
    client.bucket.return_value.blob.return_value = missing_blob
    result = differ.is_diff(
        step_name='pts_ontoma',
        config=_config_for(manifest, tmp_path),
        client=client,
    )
    assert result is True
