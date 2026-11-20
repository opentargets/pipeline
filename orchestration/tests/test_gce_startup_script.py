"""Tests for the startup script of the vm steps."""

import re
import subprocess
from pathlib import Path
from typing import Any

import pytest

from orchestration.operators.gce import GCLOUD_IMAGE, OUTPUT_FILTER, ComputeEngineRunContainerizedWorkloadSensor


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


def test_workload_output_is_filtered(script: str) -> None:
    """The container's output goes through the filter, after the container command."""
    assert f'2>&1 | {OUTPUT_FILTER}' in script


def test_output_filter_splits_progress_bars_and_keeps_the_exit_code() -> None:
    """A long carriage-return progress bar leaves no line over 64 KiB, other output survives, and the
    workload's exit code is the pipeline's."""
    bar = r"""printf ' 63%%|=====     | %d/5365 [00:11<00:06]\r' "$i" """
    workload = f"printf 'start\\n'; for i in $(seq 1 20000); do {bar}; done; printf 'end\\n'; exit 3"
    result = subprocess.run(
        ['bash', '-c', f'set -euo pipefail; ( {workload} ) 2>&1 | {OUTPUT_FILTER}'],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 3
    assert result.stdout.splitlines() == ['start', 'end']


def test_startup_script_is_valid_bash(script: str) -> None:
    """The script is assembled from string fragments, so check bash can parse it."""
    subprocess.run(['bash', '-n'], input=script, text=True, check=True)


FIXTURES = Path(__file__).parent / 'fixtures' / 'startup_scripts'
RENDER_CASES: dict[str, dict[str, Any]] = {
    'bare': {},
    'full': {
        'container_command': 'pts',
        'container_args': ['--step', 'a_step'],
        'container_env': {'A': '1', 'B': 'two'},
        'container_files': {'gs://a-bucket/a/config.yaml': '/config.yaml'},
        'container_secret_files': {'a-secret': '/secrets/a-secret'},
        'work_disk_size_gb': 100,
    },
}


@pytest.mark.parametrize('case', RENDER_CASES)
def test_startup_script_is_unchanged(case: str) -> None:
    """Fixtures were rendered before the script was split into overridable pieces."""
    rendered = ComputeEngineRunContainerizedWorkloadSensor(
        task_id='a_step',
        instance_name='a-machine',
        container_image='an-image',
        **RENDER_CASES[case],
    ).startup_script()
    assert rendered == (FIXTURES / f'{case}.sh').read_text()
