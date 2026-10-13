"""Tests for the startup script of the vm steps."""

import re
import subprocess

import pytest

from orchestration.operators.gce import GCLOUD_IMAGE, ComputeEngineRunContainerizedWorkloadSensor


@pytest.fixture
def script() -> str:
    """A startup script that copies a file and fetches a secret, so both gcloud paths render."""
    return ComputeEngineRunContainerizedWorkloadSensor(
        task_id='a_step',
        instance_name='a-machine',
        container_image='an-image',
        container_files={'gs://a-bucket/a/config.yaml': '/config.yaml'},
        container_secret_files={'a-secret': '/secrets/a-secret'},
    ).startup_script()


def test_gcloud_image_is_pinned_by_digest() -> None:
    """The helper image must not float: a tag can be repointed, a digest cannot."""
    assert re.fullmatch(r'gcr\.io/google\.com/cloudsdktool/google-cloud-cli:[\w.-]+@sha256:[0-9a-f]{64}', GCLOUD_IMAGE)


def test_startup_script_runs_gcloud_from_the_pinned_image(script: str) -> None:
    """Both the file copy and the secrets fallback use the pinned image, and nothing else."""
    images = re.findall(r'\S*cloudsdktool\S*|\S*gsutil_wrap\S*', script)
    assert images == [GCLOUD_IMAGE, GCLOUD_IMAGE]
    assert 'gcloud storage cp' in script
    assert ':latest' not in script


def test_startup_script_is_valid_bash(script: str) -> None:
    """The script is assembled from string fragments, so check bash can parse it."""
    subprocess.run(['bash', '-n'], input=script, text=True, check=True)
