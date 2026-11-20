"""Tests for the release config and the hand-off model."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from orchestration.dags.config.release import HandOff, ReleaseConfig


def hand_off(**overrides: object) -> HandOff:
    fields = {
        'release_uri': 'gs://open-targets-pipeline-runs/do/platform-2612-3',
        'run': 'do/platform-2612-3',
        'product': 'platform',
        'release': '26.12',
        'snapshots': {'clickhouse': 'do-platform-2612-3-ch', 'opensearch': 'do-platform-2612-3-os'},
        'disk_size_gb': {'clickhouse': 200, 'opensearch': 80},
        'pos_image': 'r/pos:26.12.0rc5',
        'pos_version': '26.12.0rc5',
        'clickhouse_image': 'c',
        'opensearch_image': 'o',
        'created': datetime(2026, 11, 20, tzinfo=UTC),
        'airflow_run_id': 'manual__1',
        **overrides,
    }
    return HandOff.model_validate(fields)


def test_release_config_loads_yaml():
    config = ReleaseConfig()

    assert config.backend_machine_type == 'n2d-highmem-96'
    assert config.backend_max_run_hours == 4
    assert (config.clickhouse_disk_gb, config.opensearch_disk_gb, config.work_disk_gb) == (200, 80, 200)
    assert config.pos_release_service_account == 'pos-release@open-targets-eu-dev.iam.gserviceaccount.com'
    assert config.aws_secret == 'pos-aws-credentials'  # noqa: S105
    assert config.scratch_min_pos_version == '26.12.0rc5'


def test_hand_off_round_trips():
    original = hand_off()

    assert HandOff.model_validate_json(original.model_dump_json()) == original


def test_hand_off_rejects_extra_keys():
    with pytest.raises(ValidationError):
        hand_off(surprise=1)


def test_hand_off_uri():
    assert HandOff.uri('gs://b/do/x/') == 'gs://b/do/x/release_candidate.json'
