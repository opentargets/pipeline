"""Tests for the release DAG, which publishes a release candidate."""

import copy
from collections.abc import Mapping
from typing import Any, cast
from unittest.mock import patch

import pytest
from airflow.models import DagBag
from airflow.task.trigger_rule import TriggerRule
from google.api_core.exceptions import NotFound

from orchestration.dags.config.release import HandOff, ReleaseConfig
from orchestration.dags.config.unified_pipeline import UnifiedPipelineConfig
from orchestration.operators.gce import ComputeEngineRunContainerizedWorkloadSensor, DeleteInstanceOperator
from orchestration.operators.gce_backend import ComputeEngineRunBackendSensor
from orchestration.operators.gce_disks import CreateDiskOperator

URI = 'gs://open-targets-pipeline-runs/do/platform-2612-3'
PPP_URI = 'gs://open-targets-pipeline-runs/sz/ppp-2612-1'
RUN_ID = 'manual__2026-11-20T10:00:00+00:00'
POS_SA = 'pos-release@open-targets-eu-dev.iam.gserviceaccount.com'
SMALL_STEPS = ('gcs_sync', 'aws_sync', 'bigquery', 'publish_manifest', 'scratch_cleanup')
GATED = ('aws_sync', 'bigquery', 'scratch_cleanup')
DEVICES = ('clickhouse', 'opensearch')


def _hand_off(uri: str = URI, pos_version: str = '26.12.0rc6', **over: Any) -> dict[str, Any]:
    run = '/'.join(uri.split('/')[-2:])
    base = run.replace('/', '-')
    body = {
        'release_uri': uri,
        'run': run,
        'product': 'ppp' if '/ppp-' in uri else 'platform',
        'release': '26.12',
        'snapshots': {
            'clickhouse': f'projects/open-targets-eu-dev/global/snapshots/{base}-ch',
            'opensearch': f'projects/open-targets-eu-dev/global/snapshots/{base}-os',
        },
        'disk_size_gb': {'clickhouse': 123, 'opensearch': 45},
        'pos_image': 'europe-west1-docker.pkg.dev/p/r/pos:captured',
        'pos_version': pos_version,
        'clickhouse_image': 'clickhouse/clickhouse-server:captured',
        'opensearch_image': 'opensearchproject/opensearch:captured',
        'created': '2026-11-20T12:00:00Z',
        'airflow_run_id': 'manual__run',
    }
    return {**body, **over}


def _load(scratch: bool, uri: str = URI, **hand_off: Any) -> dict[str, Any]:
    from orchestration.dags import release

    body = HandOff.model_validate(_hand_off(uri, **hand_off)).model_dump_json()
    with patch.object(release, 'Client') as client:
        blob = client.return_value.bucket.return_value.blob.return_value
        blob.download_as_text.return_value = body
        out = release.load_hand_off(uri, scratch)
    client.return_value.bucket.assert_called_once_with('open-targets-pipeline-runs')
    client.return_value.bucket.return_value.blob.assert_called_once_with(
        f'{"/".join(uri.split("/")[-2:])}/release_candidate.json'
    )
    return out


class _TI:
    """XComs keyed by `<task id>:<key>`, and this run's task states."""

    dag_id = 'release'
    run_id = RUN_ID

    def __init__(self, xcoms: dict[str, Any], states: Mapping[str, str | None] | None = None) -> None:
        self.xcoms = xcoms
        self.states = states or {}

    def get_task_states(self, dag_id: str, task_ids: list[str], run_ids: list[str]) -> dict[str, Any]:
        assert (dag_id, run_ids) == (self.dag_id, [self.run_id])
        return {self.run_id: {t: self.states[t] for t in task_ids if t in self.states}}

    def xcom_pull(self, task_ids: str, key: str = 'return_value', **_: Any) -> Any:
        return self.xcoms.get(f'{task_ids}:{key}')


def _xcoms(out: dict[str, Any]) -> dict[str, Any]:
    return {f'read_hand_off:{k}': v for k, v in out.items()}


@pytest.fixture(scope='module')
def dag():
    bag = DagBag(dag_folder='src/orchestration/dags/release.py')
    assert not bag.import_errors
    return bag.dags['release']


def _render(dag, task_id: str, field: str, out: dict[str, Any], scratch: bool = False) -> Any:
    task = dag.get_task(task_id)
    value = copy.deepcopy(getattr(task, field))
    context = {
        'params': {'release_uri': out['values']['release_uri'], 'scratch': scratch},
        'run_id': RUN_ID,
        'ti': _TI(_xcoms(out)),
    }
    return task.render_template(value, context, dag.get_template_env())


def test_params_and_run_limits(dag) -> None:
    assert dag.params.get_param('release_uri').has_value is False
    assert dag.params['scratch'] is False
    assert dag.params.get_param('scratch').schema['type'] == 'boolean'
    assert dag.max_active_runs == 1
    assert dag.tags == {'release'}


def test_task_ids(dag) -> None:
    expected = {
        'read_hand_off',
        'restore_disk_clickhouse',
        'restore_disk_opensearch',
        'run_tarballs',
        'delete_vm_tarballs',
        'delete_restored_disk_clickhouse',
        'delete_restored_disk_opensearch',
        *(f'run_{s}' for s in SMALL_STEPS),
        *(f'delete_vm_{s}' for s in SMALL_STEPS),
        *(f'gate_{s}' for s in GATED),
        'done',
    }
    assert set(dag.task_ids) == expected


def test_order(dag) -> None:
    up = {t.task_id: t.upstream_task_ids for t in dag.tasks}
    for d in DEVICES:
        assert up[f'restore_disk_{d}'] == {'read_hand_off'}
        assert up[f'delete_restored_disk_{d}'] == {f'restore_disk_{d}', 'delete_vm_tarballs'}
        assert dag.get_task(f'delete_restored_disk_{d}').trigger_rule == TriggerRule.ALL_DONE
    assert up['run_tarballs'] == {'read_hand_off', 'restore_disk_clickhouse', 'restore_disk_opensearch'}
    assert up['run_gcs_sync'] == {'read_hand_off', 'run_tarballs'}
    assert up['run_aws_sync'] == {'read_hand_off', 'gate_aws_sync', 'run_gcs_sync'}
    assert up['run_bigquery'] == {'read_hand_off', 'gate_bigquery'}
    assert up['run_publish_manifest'] == {'read_hand_off', 'run_gcs_sync', 'run_aws_sync', 'run_bigquery'}
    assert up['run_scratch_cleanup'] == {
        'read_hand_off',
        'gate_scratch_cleanup',
        'run_tarballs',
        'run_gcs_sync',
        'run_bigquery',
        'run_publish_manifest',
    }
    assert dag.get_task('run_scratch_cleanup').trigger_rule == TriggerRule.ALL_DONE_MIN_ONE_SUCCESS
    for s in GATED:
        assert up[f'gate_{s}'] == {'read_hand_off'}
        # a skipped gate skips the run and its VM delete, nothing further
        assert dag.get_task(f'gate_{s}').downstream_task_ids == {f'run_{s}', f'delete_vm_{s}'}
        assert dag.get_task(f'gate_{s}').ignore_downstream_trigger_rules is False
    for s in ('tarballs', *SMALL_STEPS):
        assert f'run_{s}' in up[f'delete_vm_{s}']
        delete = dag.get_task(f'delete_vm_{s}')
        assert isinstance(delete, DeleteInstanceOperator)
        assert delete.trigger_rule == TriggerRule.ALL_DONE
    for s in ('tarballs', 'gcs_sync', 'aws_sync', 'bigquery'):
        assert dag.get_task(f'run_{s}').trigger_rule == TriggerRule.ALL_SUCCESS
    # skipped aws_sync and bigquery must not skip the manifest
    assert dag.get_task('run_publish_manifest').trigger_rule == TriggerRule.NONE_FAILED
    dag.topological_sort()


def test_done_is_the_only_leaf_and_fails_on_any_failure(dag) -> None:
    assert [t.task_id for t in dag.leaves] == ['done']
    done = dag.get_task('done')
    # ALL_SUCCESS would mark done skipped as soon as a gated step is skipped, and a skipped leaf
    # lets the run succeed whatever fails later
    assert done.trigger_rule == TriggerRule.NONE_FAILED
    assert done.upstream_task_ids == {
        'run_tarballs',
        'delete_vm_tarballs',
        'delete_restored_disk_clickhouse',
        'delete_restored_disk_opensearch',
        *(f'run_{s}' for s in SMALL_STEPS),
        *(f'delete_vm_{s}' for s in SMALL_STEPS),
    }


def test_read_hand_off_returns_the_hand_off_and_restored_disks() -> None:
    out = _load(False)
    assert out['hand_off'] == HandOff.model_validate(_hand_off()).model_dump(mode='json')
    assert out['values'] == {
        'release_uri': URI,
        'product': 'platform',
        'pos_image': 'europe-west1-docker.pkg.dev/p/r/pos:captured',
        'clickhouse_snapshot': 'projects/open-targets-eu-dev/global/snapshots/do-platform-2612-3-ch',
        'opensearch_snapshot': 'projects/open-targets-eu-dev/global/snapshots/do-platform-2612-3-os',
        'clickhouse_disk_gb': 123,
        'opensearch_disk_gb': 45,
        'clickhouse_disk': 'pos-do-platform-2612-3-ch-r',
        'opensearch_disk': 'pos-do-platform-2612-3-os-r',
    }


def test_read_hand_off_fails_without_the_hand_off() -> None:
    from orchestration.dags import release

    with patch.object(release, 'Client') as client:
        blob = client.return_value.bucket.return_value.blob.return_value
        blob.download_as_text.side_effect = NotFound('x')
        with pytest.raises(FileNotFoundError, match=r'release_candidate\.json'):
            release.load_hand_off(URI, False)


def test_read_hand_off_fails_on_another_runs_hand_off() -> None:
    with pytest.raises(ValueError, match='release_uri'):
        _load(False, release_uri='gs://open-targets-pipeline-runs/do/platform-2612-2')


def test_read_hand_off_refuses_a_non_boolean_scratch() -> None:
    from orchestration.dags import release

    with pytest.raises(TypeError, match='scratch'):
        release.load_hand_off(URI, cast(bool, 'false'))


@pytest.mark.parametrize(
    'version',
    [
        '26.12.0rc4',
        '26.12.0rc5',
        '26.12.0.dev3.pos-phase3.abc',
        '26.12.0+local',
        'captured',
        '25.12.0rc9',
        '25.12.0',
        '26.12',
    ],
)
def test_scratch_refuses_old_or_unparsable_pos_versions(version: str) -> None:
    assert ReleaseConfig().scratch_min_pos_version == '26.12.0rc6'
    with pytest.raises(ValueError, match='scratch'):
        _load(True, pos_version=version)
    assert _load(False, pos_version=version)['env'] == {}


@pytest.mark.parametrize('version', ['26.12.0rc6', '26.12.0rc10', '26.12.0', '26.12.1rc1', '27.1.0rc1'])
def test_scratch_accepts_new_pos_versions(version: str) -> None:
    assert _load(True, pos_version=version)['env'] == {'POS_SCRATCH': '1'}


@pytest.mark.parametrize(
    ('uri', 'scratch', 'runs'),
    [
        (URI, False, {'bigquery', 'aws_sync'}),
        (URI, True, {'bigquery', 'scratch_cleanup'}),
        (PPP_URI, False, set()),
        (PPP_URI, True, set()),
    ],
)
def test_gates(uri: str, scratch: bool, runs: set[str]) -> None:
    from orchestration.dags import release

    gates = _load(scratch, uri)['gates']
    assert set(gates) == set(GATED)
    assert {s for s, on in gates.items() if on} == runs
    for step in GATED:
        gate = release.gate_open(gates, step)
        assert gate is (step in runs)


def _pos_vms(dag) -> list[ComputeEngineRunContainerizedWorkloadSensor]:
    return [t for t in dag.tasks if isinstance(t, ComputeEngineRunContainerizedWorkloadSensor)]


def test_every_step_runs_a_pos_vm(dag) -> None:
    assert {t.task_id for t in _pos_vms(dag)} == {f'run_{s}' for s in ('tarballs', *SMALL_STEPS)}


@pytest.mark.parametrize('scratch', [False, True])
def test_pos_vms_run_the_hand_off_image_as_pos_release(dag, scratch: bool) -> None:
    out = _load(scratch)
    for t in _pos_vms(dag):
        assert t.vm_service_account == POS_SA
        assert t.deferrable is True
        assert _render(dag, t.task_id, 'container_image', out, scratch) == out['values']['pos_image']
        env = _render(dag, t.task_id, 'container_env', out, scratch)
        assert ('POS_SCRATCH' in env) is scratch
        if scratch:
            assert env['POS_SCRATCH'] == '1'
        step = t.task_id.removeprefix('run_')
        labels = _render(dag, t.task_id, 'labels', out, scratch)
        assert (labels['tool'], labels['step'], labels['product']) == ('pos', step, 'platform')
        pool = '16' if step == 'tarballs' else '8'
        args = ' '.join(_render(dag, t.task_id, 'container_args', out, scratch))
        assert args == f'-s {step} -r {URI} -w /mnt/disks/work -p {pool}'


def test_labels_carry_the_ppp_product(dag) -> None:
    out = _load(False, PPP_URI)
    for t in _pos_vms(dag):
        assert _render(dag, t.task_id, 'labels', out)['product'] == 'ppp'


def test_small_steps(dag) -> None:
    rc = ReleaseConfig()
    up = UnifiedPipelineConfig()
    for s in SMALL_STEPS:
        t = dag.get_task(f'run_{s}')
        assert type(t) is ComputeEngineRunContainerizedWorkloadSensor
        assert t.machine_type == rc.small_step_machine_type
        assert t.work_disk_size_gb == rc.small_step_work_disk_gb
        assert t.boot_disk_size_gb == rc.boot_disk_gb
        assert t.source_image == up.cos_image


@pytest.mark.parametrize('scratch', [False, True])
@pytest.mark.parametrize('step', ['aws_sync', 'publish_manifest'])
def test_aws_credentials(dag, step: str, scratch: bool) -> None:
    out = _load(scratch)
    t = dag.get_task(f'run_{step}')
    secrets = _render(dag, t.task_id, 'container_secret_files', out, scratch)
    env = _render(dag, t.task_id, 'container_env', out, scratch)
    if scratch:
        assert secrets == {}
        assert 'AWS_SHARED_CREDENTIALS_FILE' not in env
    else:
        assert secrets == {'pos-aws-credentials': '/secrets/aws/credentials'}
        assert env['AWS_SHARED_CREDENTIALS_FILE'] == '/secrets/aws/credentials'
    assert ('POS_SCRATCH' in env) is scratch


@pytest.mark.parametrize('scratch', [False, True])
def test_scratch_manifest_vm_fetches_no_secret(dag, scratch: bool) -> None:
    out = _load(scratch)
    t = copy.deepcopy(dag.get_task('run_publish_manifest'))
    t.render_template_fields(
        {'params': {'release_uri': URI, 'scratch': scratch}, 'run_id': RUN_ID, 'ti': _TI(_xcoms(out))},
        dag.get_template_env(),
    )
    assert ('"pos-aws-credentials"' in t.startup_script()) is not scratch


def test_other_steps_have_no_aws_credentials(dag) -> None:
    out = _load(False)
    for s in ('tarballs', 'gcs_sync', 'bigquery', 'scratch_cleanup'):
        t = dag.get_task(f'run_{s}')
        assert not t.container_secret_files
        assert 'AWS_SHARED_CREDENTIALS_FILE' not in _render(dag, t.task_id, 'container_env', out)


def test_tarballs_reads_the_restored_disks(dag) -> None:
    rc = ReleaseConfig()
    out = _load(False)
    t = cast(ComputeEngineRunBackendSensor, dag.get_task('run_tarballs'))
    assert isinstance(t, ComputeEngineRunBackendSensor)
    assert not t.servers
    assert t.read_only is True
    assert t.machine_type == rc.tarballs_machine_type
    assert t.work_disk_size_gb == rc.small_step_work_disk_gb
    assert _render(dag, 'run_tarballs', 'data_disks', out) == {
        'clickhouse': 'pos-do-platform-2612-3-ch-r',
        'opensearch': 'pos-do-platform-2612-3-os-r',
    }


def test_restored_disks_come_from_the_hand_off_snapshots(dag) -> None:
    out = _load(False)
    for d, short, size in (('clickhouse', 'ch', '123'), ('opensearch', 'os', '45')):
        t = dag.get_task(f'restore_disk_{d}')
        assert isinstance(t, CreateDiskOperator)
        assert _render(dag, t.task_id, 'disk_name', out) == f'pos-do-platform-2612-3-{short}-r'
        assert str(_render(dag, t.task_id, 'size_gb', out)) == size
        assert _render(dag, t.task_id, 'source_snapshot', out) == (
            f'projects/open-targets-eu-dev/global/snapshots/do-platform-2612-3-{short}'
        )
        labels = dict(_render(dag, t.task_id, 'labels', out))
        assert (labels['tool'], labels['step'], labels['product']) == ('pos', 'tarballs', 'platform')
        assert labels['run'] == 'manual__2026-11-20t10-00-00-00-00'


RESTORED = {
    'restore_disk_clickhouse:disk': 'pos-do-platform-2612-3-ch-r',
    'restore_disk_opensearch:disk': 'pos-do-platform-2612-3-os-r',
}


@pytest.mark.parametrize(
    ('states', 'deleted'),
    [
        (
            {'read_hand_off': 'success', 'restore_disk_clickhouse': 'success', 'restore_disk_opensearch': 'failed'},
            ['pos-do-platform-2612-3-ch-r', 'pos-do-platform-2612-3-os-r'],
        ),
        # a cleared run whose hand-off read failed keeps the old XComs
        ({'read_hand_off': 'failed', 'restore_disk_clickhouse': None, 'restore_disk_opensearch': None}, []),
        (
            {'read_hand_off': 'success', 'restore_disk_clickhouse': 'upstream_failed',
             'restore_disk_opensearch': None},
            [],
        ),
    ],
)  # fmt: skip
def test_restored_disk_deletes_remove_only_disks_this_try_created(states, deleted) -> None:
    from orchestration.dags import release

    with patch.object(release, 'delete_disk') as delete:
        for d in DEVICES:
            release.cleanup_restored_disk(_TI(RESTORED, states), d)
    assert [c.args[2] for c in delete.call_args_list] == deleted


def test_a_final_version_sorts_after_its_rcs() -> None:
    from orchestration.dags import release

    keys = [release.pos_version_key(v) for v in ('26.12.0rc99', '26.12.0', '26.12.1rc1')]
    assert None not in keys
    assert keys == sorted(cast(list[tuple[int, ...]], keys))
    assert len(set(keys)) == 3
    assert release.pos_version_key('26.12.0.dev3.pos-phase3.abc') is None


def _simulate(dag, gates: dict[str, bool], failed: frozenset[str] | set[str] = frozenset()) -> dict[str, str]:
    """Final task states, by the trigger-rule outcomes of `trigger_rule_dep` once every upstream is done.

    Tasks in `failed` fail when they run; every other task succeeds. A closed gate skips its direct
    downstream, as `ShortCircuitOperator(ignore_downstream_trigger_rules=False)`.
    """
    states: dict[str, str] = {}
    gate_skipped: set[str] = set()
    for t in dag.topological_sort():
        up = [states[u] for u in t.upstream_task_ids]
        bad = any(s in {'failed', 'upstream_failed'} for s in up)
        rule = t.trigger_rule
        if t.task_id in gate_skipped:
            state = 'skipped'
        elif rule == TriggerRule.ALL_SUCCESS and bad:
            state = 'upstream_failed'
        elif rule == TriggerRule.ALL_SUCCESS and 'skipped' in up:
            state = 'skipped'
        elif rule == TriggerRule.NONE_FAILED and bad:
            state = 'upstream_failed'
        elif rule == TriggerRule.ALL_DONE_MIN_ONE_SUCCESS and 'skipped' in up:
            state = 'skipped'
        elif rule == TriggerRule.ALL_DONE_MIN_ONE_SUCCESS and 'success' not in up:
            state = 'upstream_failed'
        else:
            assert rule in {
                TriggerRule.ALL_SUCCESS, TriggerRule.ALL_DONE, TriggerRule.NONE_FAILED,
                TriggerRule.ALL_DONE_MIN_ONE_SUCCESS,
            }  # fmt: skip
            state = 'failed' if t.task_id in failed else 'success'
            if state == 'success' and t.task_id.startswith('gate_') and not gates[t.task_id.removeprefix('gate_')]:
                gate_skipped |= t.downstream_task_ids
        states[t.task_id] = state
    return states


def _ran(states: dict[str, str]) -> set[str]:
    return {t.removeprefix('run_') for t, s in states.items() if t.startswith('run_') and s in {'success', 'failed'}}


@pytest.mark.parametrize(
    ('uri', 'scratch', 'runs'),
    [
        (URI, False, {'tarballs', 'gcs_sync', 'aws_sync', 'bigquery', 'publish_manifest'}),
        (URI, True, {'tarballs', 'gcs_sync', 'bigquery', 'publish_manifest', 'scratch_cleanup'}),
        (PPP_URI, False, {'tarballs', 'gcs_sync', 'publish_manifest'}),
        (PPP_URI, True, {'tarballs', 'gcs_sync', 'publish_manifest'}),
    ],
)
def test_a_successful_run_runs_its_steps_and_succeeds(dag, uri: str, scratch: bool, runs: set[str]) -> None:
    states = _simulate(dag, _load(scratch, uri)['gates'])
    assert _ran(states) == runs
    assert states['done'] == 'success'
    assert {s for s in GATED if states[f'run_{s}'] == 'skipped'} == set(GATED) - runs
    assert all(states[f'delete_vm_{s}'] == ('success' if s in runs else 'skipped') for s in SMALL_STEPS)
    if 'scratch_cleanup' in runs:
        assert 'run_publish_manifest' in dag.get_task('run_scratch_cleanup').upstream_task_ids


@pytest.mark.parametrize('uri', [URI, PPP_URI])
@pytest.mark.parametrize('scratch', [False, True])
@pytest.mark.parametrize('step', ['run_tarballs', 'restore_disk_opensearch', 'run_gcs_sync', 'run_publish_manifest'])
def test_any_failure_fails_the_run(dag, uri: str, scratch: bool, step: str) -> None:
    states = _simulate(dag, _load(scratch, uri)['gates'], {step})
    assert states['done'] == 'upstream_failed'
    assert states['delete_vm_tarballs'] == 'success'
    assert states['delete_restored_disk_clickhouse'] == 'success'


def test_a_failed_scratch_run_still_drops_its_datasets(dag) -> None:
    states = _simulate(dag, _load(True)['gates'], {'run_gcs_sync'})
    assert states['run_publish_manifest'] == 'upstream_failed'
    assert states['run_scratch_cleanup'] == 'success'
    assert states['delete_vm_scratch_cleanup'] == 'success'
    assert states['done'] == 'upstream_failed'


@pytest.mark.parametrize('step', ['run_bigquery', 'run_aws_sync'])
def test_a_failed_platform_step_fails_the_run(dag, step: str) -> None:
    states = _simulate(dag, _load(False)['gates'], {step})
    assert states['run_publish_manifest'] == 'upstream_failed'
    assert states['run_scratch_cleanup'] == 'skipped'
    assert states['done'] == 'upstream_failed'


def test_no_cleanup_without_a_hand_off(dag) -> None:
    states = _simulate(dag, {}, {'read_hand_off'})
    assert states['run_scratch_cleanup'] == 'upstream_failed'
    assert _ran(states) == set()
    assert states['done'] == 'upstream_failed'
