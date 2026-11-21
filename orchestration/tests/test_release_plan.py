"""Tests for the per-side build/reuse plan."""

import json
from dataclasses import asdict
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from airflow.sdk.exceptions import AirflowException
from google.api_core.exceptions import NotFound

from orchestration.utils import release
from orchestration.utils.release import candidate_names, plan_candidate

URI = 'gs://bucket/releases/do/platform-2612-3'
CH, OS = 'do-platform-2612-3-ch', 'do-platform-2612-3-os'
OS_DISK = 'pos-do-platform-2612-3-os'
PROV = {'pos_image': 'p', 'server_image': 'ch-img', 'airflow_run_id': 'r'}
OS_PROV = {**PROV, 'server_image': 'os-img'}


def _names() -> dict[str, Any]:
    return {
        **asdict(candidate_names(URI)),
        'pos_image': 'p',
        'clickhouse_image': 'ch-img',
        'opensearch_image': 'os-img',
        'clickhouse_disk_gb': 170,
        'opensearch_disk_gb': 80,
    }


def _snap(status='READY', description=None, size=170):
    return SimpleNamespace(
        status=status,
        description=json.dumps(PROV) if description is None else description,
        disk_size_gb=size,
    )


@pytest.fixture
def env():
    """Patched clients; fill `.snaps`, `.disks`, `.blob`."""
    state = SimpleNamespace(snaps={}, disks=set(), blob=False)

    def sget(**kw):
        if kw['snapshot'] in state.snaps:
            return state.snaps[kw['snapshot']]
        raise NotFound('x')

    def dget(**kw):
        if kw['disk'] in state.disks:
            return MagicMock()
        raise NotFound('x')

    with (
        patch.object(release.compute_v1, 'SnapshotsClient') as sc,
        patch.object(release.compute_v1, 'DisksClient') as dc,
        patch.object(release, 'Client') as cl,
    ):
        sc.return_value.get.side_effect = sget
        dc.return_value.get.side_effect = dget
        cl.return_value.bucket.return_value.blob.return_value.exists.side_effect = lambda: state.blob
        yield state


def _plan(names=None):
    return plan_candidate(names or _names(), 'proj', 'zone')


def test_plan_builds_missing_sides(env) -> None:
    assert _plan() == {
        'clickhouse': {'action': 'build', 'disk_size_gb': 170},
        'opensearch': {'action': 'build', 'disk_size_gb': 80},
    }


def test_plan_reuses_matching_snapshot(env) -> None:
    env.snaps[CH] = _snap()
    assert _plan() == {
        'clickhouse': {'action': 'reuse', 'disk_size_gb': 170},
        'opensearch': {'action': 'build', 'disk_size_gb': 80},
    }


def test_plan_records_reused_disk_size(env) -> None:
    env.snaps[CH] = _snap(size=170)
    assert _plan({**_names(), 'clickhouse_disk_gb': 200})['clickhouse']['disk_size_gb'] == 170


def test_plan_reuses_both(env) -> None:
    env.snaps[CH] = _snap()
    env.snaps[OS] = _snap(description=json.dumps(OS_PROV), size=80)
    plan = _plan()
    assert {s: p['action'] for s, p in plan.items()} == {'clickhouse': 'reuse', 'opensearch': 'reuse'}


def test_plan_fails_on_other_pos_image(env) -> None:
    env.snaps[CH] = _snap(description=json.dumps({**PROV, 'pos_image': 'old'}))
    with pytest.raises(AirflowException, match=rf'{CH}.*delete'):
        _plan()


def test_plan_fails_on_other_server_image(env) -> None:
    env.snaps[CH] = _snap(description=json.dumps({**PROV, 'server_image': 'old'}))
    with pytest.raises(AirflowException, match=rf'{CH}.*delete'):
        _plan()


@pytest.mark.parametrize('description', ['', 'x', '{"pos_image": "p"}', '[1]', '"s"'])
def test_plan_fails_on_unreadable_provenance(env, description) -> None:
    env.snaps[CH] = _snap(description=description)
    with pytest.raises(AirflowException, match=rf'{CH}.*no provenance.*delete'):
        _plan()


@pytest.mark.parametrize('status', ['CREATING', 'UPLOADING'])
def test_plan_tells_to_wait_for_a_snapshot_being_created(env, status) -> None:
    env.snaps[CH] = _snap(status=status)
    with pytest.raises(AirflowException, match=rf'{CH}.*wait') as e:
        _plan()
    assert 'delete' not in str(e.value)


@pytest.mark.parametrize('status', ['FAILED', 'DELETING'])
def test_plan_tells_to_delete_a_broken_snapshot(env, status) -> None:
    env.snaps[CH] = _snap(status=status)
    with pytest.raises(AirflowException, match=rf'{CH}.*delete'):
        _plan()


def test_plan_fails_on_hand_off(env) -> None:
    env.blob = True
    with pytest.raises(AirflowException, match=r'release_candidate\.json'):
        _plan()


def test_plan_fails_on_leftover_disk_of_a_side_to_build(env) -> None:
    env.disks.add(OS_DISK)
    with pytest.raises(AirflowException, match=OS_DISK):
        _plan()


def test_plan_ignores_disk_of_a_reused_side(env) -> None:
    env.disks.add(OS_DISK)
    env.snaps[CH] = _snap()
    env.snaps[OS] = _snap(description=json.dumps(OS_PROV), size=80)
    assert _plan()['opensearch']['action'] == 'reuse'


def test_plan_fails_on_leftover_clickhouse_build_disk(env) -> None:
    env.disks.add('pos-do-platform-2612-3-ch-build')
    with pytest.raises(AirflowException, match='ch-build'):
        _plan()
