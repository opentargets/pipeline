"""Tests for the staging DAGs."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml
from airflow.models import DagBag
from airflow.task.trigger_rule import TriggerRule
from pydantic import ValidationError

from orchestration.dags.config.staging import DATAPROC_INIT_SCRIPT, HAIL_JAR, PPS_CONFIG_PATH, StagingDagConfig
from orchestration.models.staging_dag import StagingDagConfigModel
from orchestration.utils.common import GCP_PROJECT_GENETICS, GCP_ZONE
from orchestration.utils.staging import build_staging_dag

DAGS_PATH = Path(__file__).parent.parent / 'src' / 'orchestration' / 'dags'
STAGING_DAG_DIRS = sorted(p.parent for p in DAGS_PATH.glob('*/config.yaml'))

BATCH_STEP: dict[str, Any] = {
    'google_batch_index_specs': {
        'pointer': 'finemapping',
        'max_task_count': 10,
        'generator_specs': {'generator_options': {'collected_loci_path': 'gs://bucket/loci'}},
    },
    'google_batch': {
        'job': {
            'task_group': {
                'parallelism': 10,
                'task_environments': {'environments': []},
                'task_config': {
                    'instance_resource_spec': {'cpu_milli': 1000, 'memory_mib': 1000, 'boot_disk_mib': 10000},
                    'runnable_spec': {'image_uri': 'gentropy:latest', 'inline_commands': ['-c', 'true']},
                },
            },
            'allocation': {'instance': {'machine_type': 'n2-standard-4'}},
            'logs': {},
        },
    },
}


VM_STEP: dict[str, Any] = {'vm': {'machine_type': 'n1-standard-4'}}

DELTA_JARS = ['io.delta:delta-spark_2.12:3.2.1', 'io.delta:delta-storage:3.2.1']


def dag_config(**overrides: Any) -> dict[str, Any]:
    return {
        'env': 'prod',
        'environments': [
            {'name': 'prod', 'scratchpad': {'workers': '4', 'gwas_catalog_top_hits_path': 'gs://prod'}},
            {'name': 'test', 'scratchpad': {'workers': '2', 'gwas_catalog_top_hits_path': 'gs://test'}},
        ],
        'description': 'a staging dag used in the tests of the staging dags',
        'staging_bucket': 'gs://staging/',
        'pps_version': '26.9.0',
        'infrastructure': {'clusters': {'cluster': {'num_workers': '{{workers}}'}}},
        'steps': {'gwas_catalog_top_hits_ingestion': {'cluster': 'cluster'}},
    } | overrides


def write_dag(tmp_path: Path, config: dict[str, Any], dag_id: str = 'test_staging_dag') -> Path:
    dag_dir = tmp_path / dag_id
    dag_dir.mkdir()
    (dag_dir / 'config.yaml').write_text(yaml.dump(config, allow_unicode=True))
    return dag_dir


class TestStagingDagConfigModel:
    @staticmethod
    def validate(config: dict[str, Any]) -> StagingDagConfigModel:
        # the model validates rendered configs
        config = yaml.safe_load(yaml.dump(config).replace('{{workers}}', '4'))
        return StagingDagConfigModel.model_validate(config)

    def test_valid_config(self) -> None:
        model = self.validate(
            dag_config(
                steps={
                    'a': {'cluster': 'cluster'},
                    'b': {'cluster': 'cluster', 'depends_on': ['a'], 'execution_timeout': 3600},
                    'c': {**BATCH_STEP, 'depends_on': ['b']},
                    'd': {**VM_STEP, 'depends_on': ['c']},
                }
            )
        )

        assert [s.kind for s in model.steps.values()] == ['cluster', 'cluster', 'batch', 'vm']
        vm = model.steps['d'].vm
        assert vm is not None
        assert vm.work_disk_size_gb == 0
        assert model.scratchpad == {'workers': '4', 'gwas_catalog_top_hits_path': 'gs://prod'}

    def test_batch_only_dag_needs_no_pps(self) -> None:
        config = dag_config(steps={'a': BATCH_STEP}, infrastructure={})
        del config['staging_bucket'], config['pps_version']

        assert self.validate(config).pps_version is None

    def test_staging_clusters_use_default_service_account(self) -> None:
        cluster = self.validate(dag_config()).infrastructure.clusters['cluster']

        assert cluster.service_account is None
        assert cluster.internal_ip_only is False

    @pytest.mark.parametrize(
        ('overrides', 'message'),
        [
            ({'env': 'staging'}, "Input should be 'prod' or 'test'"),
            ({'env': 'test', 'environments': [{'name': 'prod'}]}, 'environment test is not defined'),
            ({'steps': {'a': {}}}, 'exactly one of `cluster`'),
            ({'steps': {'a': {**BATCH_STEP, 'cluster': 'cluster'}}}, 'exactly one of `cluster`'),
            ({'steps': {'a': {**VM_STEP, 'cluster': 'cluster'}}}, 'exactly one of `cluster`'),
            (
                {'steps': {'a': {**VM_STEP, 'google_batch_index_specs': BATCH_STEP['google_batch_index_specs']}}},
                'only valid for Batch steps',
            ),
            ({'steps': {'a': {'vm': {}}}}, 'machine_type'),
            ({'steps': {'a': {'vm': {'machine_type': 'n1-standard-4', 'work_disk_size_gb': -1}}}}, 'greater than'),
            ({'steps': {'a': {'vm': {'machine_type': 'n1-standard-4', 'scopes': []}}}}, 'Extra inputs'),
            ({'steps': {'a' * 55: VM_STEP}}, 'too long to name its VM'),
            ({'pps_version': None, 'steps': {'a': VM_STEP}}, 'a dag with pps steps needs pps_version'),
            ({'infrastructure': {'clusters': {'cluster': {'jars': {'hail': ['a:b:1']}}}}}, '`hail` lists no jars'),
            ({'infrastructure': {'clusters': {'cluster': {'jars': {'delta': []}}}}}, '`delta` needs the jars'),
            ({'infrastructure': {'clusters': {'cluster': {'jars': {'delta': ['io.delta']}}}}}, 'should match pattern'),
            ({'infrastructure': {'clusters': {'cluster': {'jars': {'spark_nlp': ['a:b:1']}}}}}, "'delta', 'bgzip'"),
            ({'steps': {'a': {'google_batch': BATCH_STEP['google_batch']}}}, 'needs `google_batch_index_specs`'),
            ({'steps': {'a': {'cluster': 'missing'}}}, 'unknown cluster'),
            ({'pps_version': None}, 'a dag with pps steps needs pps_version'),
            ({'staging_bucket': None, 'pps_version': None}, 'needs staging_bucket and pps_version'),
            ({'steps': {'a': {'cluster': 'cluster', 'depends_on': ['missing']}}}, 'unknown steps'),
            (
                {
                    'steps': {
                        'a': {'cluster': 'cluster', 'depends_on': ['b']},
                        'b': {'cluster': 'cluster', 'depends_on': ['a']},
                    }
                },
                'steps form a cycle',
            ),
        ],
    )
    def test_invalid_config(self, overrides: dict[str, Any], message: str) -> None:
        with pytest.raises(ValidationError, match=message):
            self.validate(dag_config(**overrides))


class TestStagingDagConfig:
    def test_renders_with_the_selected_environment(self, tmp_path: Path) -> None:
        config = StagingDagConfig(write_dag(tmp_path, dag_config(env='test')))

        assert config.dag_id == 'test_staging_dag'
        assert config.dag.infrastructure.clusters['cluster'].num_workers == 2

    def test_raises_on_placeholder_missing_from_the_scratchpad(self, tmp_path: Path) -> None:
        dag_dir = write_dag(tmp_path, dag_config(staging_bucket='{{bucket}}'))

        with pytest.raises(ValueError, match=r'\{\{bucket\}\}'):
            StagingDagConfig(dag_dir)

    def test_step_config_is_the_pps_config_of_the_step(self, tmp_path: Path) -> None:
        config = StagingDagConfig(write_dag(tmp_path, dag_config()))

        step_config = config.step_config('gwas_catalog_top_hits_ingestion')

        assert list(step_config['steps']) == ['gwas_catalog_top_hits_ingestion']
        assert step_config['scratchpad']['gwas_catalog_top_hits_path'] == 'gs://prod'
        assert step_config['scratchpad']['gwas_catalog_input_path'] == 'input/gwas_catalog'
        assert step_config['work_path'] == 'work'

    def test_step_config_raises_for_step_missing_from_pps(self, tmp_path: Path) -> None:
        config = StagingDagConfig(write_dag(tmp_path, dag_config()))

        with pytest.raises(ValueError, match='step missing_step of dag test_staging_dag is not defined'):
            config.step_config('missing_step')

    @pytest.mark.parametrize(('disk', 'work_path'), [(0, 'work'), (100, '/mnt/disks/work')])
    def test_vm_step_works_on_its_work_disk(self, tmp_path: Path, disk: int, work_path: str) -> None:
        vm_step = {'vm': {'machine_type': 'n1-standard-4', 'work_disk_size_gb': disk}}
        config = StagingDagConfig(write_dag(tmp_path, dag_config(steps={'gwas_catalog_release': vm_step})))

        assert config.step_config('gwas_catalog_release')['work_path'] == work_path

    def test_step_needs_the_jars_of_the_features_it_uses(self, tmp_path: Path) -> None:
        # gnomad_ld_index starts hail
        dag_dir = write_dag(tmp_path, dag_config(steps={'gnomad_ld_index': {'cluster': 'cluster'}}))

        with pytest.raises(ValueError, match='gnomad_ld_index of dag test_staging_dag uses hail, but its cluster'):
            StagingDagConfig(dag_dir)

    def test_cluster_provides_its_jars(self, tmp_path: Path) -> None:
        cluster = {
            'num_workers': '{{workers}}',
            'properties': {'spark:spark.jars': 'gs://other/some.jar'},
            'jars': {'delta': DELTA_JARS, 'hail': []},
        }
        config = StagingDagConfig(write_dag(tmp_path, dag_config(infrastructure={'clusters': {'cluster': cluster}})))
        properties = config.cluster_config('cluster').properties or {}

        assert config.staged_jars('cluster') == [
            (
                'https://repo1.maven.org/maven2/io/delta/delta-spark_2.12/3.2.1/delta-spark_2.12-3.2.1.jar',
                'gs://staging/jars/delta-spark_2.12-3.2.1.jar',
            ),
            (
                'https://repo1.maven.org/maven2/io/delta/delta-storage/3.2.1/delta-storage-3.2.1.jar',
                'gs://staging/jars/delta-storage-3.2.1.jar',
            ),
        ]
        assert properties['spark:spark.jars'].split(',') == [
            'gs://other/some.jar',
            'gs://staging/jars/delta-spark_2.12-3.2.1.jar',
            'gs://staging/jars/delta-storage-3.2.1.jar',
            HAIL_JAR,
        ]
        assert properties['spark:spark.driver.extraClassPath'] == HAIL_JAR
        assert properties['spark:spark.executor.extraClassPath'] == './hail-all-spark.jar'

    def test_cluster_without_jars_keeps_its_properties(self, tmp_path: Path) -> None:
        config = StagingDagConfig(write_dag(tmp_path, dag_config()))

        assert config.staged_jars('cluster') == []
        assert config.cluster_config('cluster').properties is None

    def test_uris_and_cluster_config(self, tmp_path: Path) -> None:
        config = StagingDagConfig(write_dag(tmp_path, dag_config()))
        cluster = config.cluster_config('cluster')

        assert config.config_uri('a') == 'gs://staging/test_staging_dag/26.9.0/config/a.yaml'
        assert cluster.metadata == {'PPS_REF': 'pps%40v26.9.0'}
        assert cluster.init_actions_uris == [f'gs://staging/test_staging_dag/26.9.0/bin/{DATAPROC_INIT_SCRIPT}']
        assert config.pps_image == 'europe-west1-docker.pkg.dev/open-targets-eu-dev/pipeline/pps:26.9.0'


@pytest.mark.parametrize('dag_dir', STAGING_DAG_DIRS, ids=lambda p: p.name)
def test_pps_steps_of_staging_dags_exist_in_pps(dag_dir: Path) -> None:
    config = StagingDagConfig(dag_dir)

    for step_name, step in config.steps.items():
        if step.kind != 'batch':
            assert config.step_config(step_name), f'{step_name} missing from {PPS_CONFIG_PATH}'


def tasks_of(dag: Any) -> dict[str, Any]:
    """The tasks of a DAG, by task id without its task group prefixes."""
    return {t.task_id.rsplit('.', 1)[-1]: t for t in dag.tasks}


def upstream(task: Any) -> set[str]:
    return {t.rsplit('.', 1)[-1] for t in task.upstream_task_ids}


def test_jars_are_staged_before_their_cluster_is_created(tmp_path: Path) -> None:
    cluster = {'num_workers': '{{workers}}', 'jars': {'delta': DELTA_JARS}}
    config = dag_config(
        infrastructure={'clusters': {'cluster': cluster}},
        steps={
            'gwas_catalog_top_hits_ingestion': {'cluster': 'cluster'},
            'gwas_catalog_top_hits_ld_clumping': {'cluster': 'cluster'},
        },
    )
    tasks = {t.task_id: t for t in build_staging_dag(write_dag(tmp_path, config)).tasks}
    stages = {t for t in tasks if t.startswith('stage_jar_')}

    assert stages == {'stage_jar_delta-spark-2-12-3-2-1-jar', 'stage_jar_delta-storage-3-2-1-jar'}
    assert all((tasks[s].retries or 0) > 0 for s in stages)
    for step in ('gwas_catalog_top_hits_ingestion', 'gwas_catalog_top_hits_ld_clumping'):
        assert stages <= tasks[f'{step}.create_cluster_cluster'].upstream_task_ids


def test_no_jars_no_staging(tmp_path: Path) -> None:
    tasks = tasks_of(build_staging_dag(write_dag(tmp_path, dag_config())))

    assert not [t for t in tasks if t.startswith('stage_jar_')]


def test_steps_are_task_groups(tmp_path: Path) -> None:
    config = dag_config(
        steps={'gwas_catalog_top_hits_ingestion': {'cluster': 'cluster'}, 'gwas_catalog_release': VM_STEP}
    )
    dag = build_staging_dag(write_dag(tmp_path, config))
    groups = {t.task_id.split('.')[0] for t in dag.tasks if '.' in t.task_id}

    assert groups == {'gwas_catalog_top_hits_ingestion', 'gwas_catalog_release'}
    assert {t.task_id for t in dag.tasks if '.' not in t.task_id} == {'delete_cluster_cluster'}


def test_batch_step_is_mapped_over_its_index(tmp_path: Path) -> None:
    config = dag_config(
        steps={
            'gwas_catalog_top_hits_ingestion': {'cluster': 'cluster'},
            'susie_finemapping': {**BATCH_STEP, 'depends_on': ['gwas_catalog_top_hits_ingestion']},
        }
    )
    tasks = tasks_of(build_staging_dag(write_dag(tmp_path, config)))

    assert upstream(tasks['generate_manifest_susie_finemapping']) == {'end_gwas_catalog_top_hits_ingestion'}
    assert upstream(tasks['run_susie_finemapping']) == {'generate_manifest_susie_finemapping'}
    assert tasks['run_susie_finemapping'].is_mapped
    assert upstream(tasks['end_susie_finemapping']) == {'run_susie_finemapping'}


def test_vm_step(tmp_path: Path) -> None:
    config = dag_config(steps={'gwas_catalog_release': {**VM_STEP, 'execution_timeout': 7200}})
    tasks = tasks_of(build_staging_dag(write_dag(tmp_path, config)))
    upload, run, delete, end = (
        tasks[f'{t}_gwas_catalog_release'] for t in ('upload_config', 'run', 'delete_vm', 'end')
    )

    assert upstream(run) == {'upload_config_gwas_catalog_release'}
    assert upstream(delete) == {'run_gwas_catalog_release'}
    assert upstream(end) == {'run_gwas_catalog_release', 'delete_vm_gwas_catalog_release'}
    # the vm is deleted even when the step fails, which still fails the dag through `end`
    assert delete.trigger_rule == TriggerRule.NONE_SKIPPED
    assert end.trigger_rule == TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS

    assert (run.project, run.zone) == (delete.project_id, delete.zone) == (GCP_PROJECT_GENETICS, GCP_ZONE)
    assert run.instance_name == delete.resource_id == 'up-gwas-catalog-release-{{ run_id | strhash }}'
    assert run.container_image == 'europe-west1-docker.pkg.dev/open-targets-eu-dev/pipeline/pps:26.9.0'
    assert run.container_args == ['-s', 'gwas_catalog_release', '-c', '/config.yaml']
    assert run.container_files == {str(upload.dst_uri): '/config.yaml'}
    assert (run.machine_type, run.work_disk_size_gb) == ('n1-standard-4', 0)
    assert run.execution_timeout.total_seconds() == 7200
    assert run.deferrable


def test_gwas_catalog_top_hits_dag(dag_bag: DagBag) -> None:
    tasks = tasks_of(dag_bag.dags['gwas_catalog_top_hits'])
    steps = ['gwas_catalog_top_hits_ingestion', 'gwas_catalog_top_hits_ld_clumping', 'gwas_catalog_top_hits_pics']
    cluster = 'otg-tophit-gwascatalog'

    for step in steps:
        assert upstream(tasks[f'run_{step}']) == {f'create_cluster_{cluster}'}
        assert f'end_{step}' in upstream(tasks[f'delete_cluster_{cluster}'])
    assert upstream(tasks[f'upload_config_{steps[1]}']) == {f'end_{steps[0]}'}
    assert upstream(tasks[f'upload_config_{steps[2]}']) == {f'end_{steps[1]}'}


def test_gwas_catalog_susie_finemapping_dag(dag_bag: DagBag) -> None:
    dag = dag_bag.dags['gwas_catalog_sumstats_susie_finemapping']
    tasks = tasks_of(dag)
    finemapping, qc = 'gwas_catalog_susie_finemapping', 'gwas_catalog_susie_credible_set_qc'

    assert tasks[f'run_{finemapping}'].is_mapped
    assert upstream(tasks[f'run_{finemapping}']) == {f'generate_manifest_{finemapping}'}
    # the qc step, and so the creation of its cluster, only start once the batch job is done
    assert upstream(tasks[f'upload_config_{qc}']) == {f'end_{finemapping}'}


def test_gwas_catalog_release_dag(dag_bag: DagBag) -> None:
    tasks = tasks_of(dag_bag.dags['gwas_catalog_release'])

    assert set(tasks) == {f'{t}_gwas_catalog_release' for t in ('upload_config', 'run', 'delete_vm', 'end')}
    assert tasks['run_gwas_catalog_release'].machine_type == 'n1-standard-4'


def test_finngen_ukb_mvp_meta_dag_stages_the_bgzip_jars(dag_bag: DagBag) -> None:
    tasks = tasks_of(dag_bag.dags['finngen_ukb_mvp_meta'])

    assert {t for t in tasks if t.startswith('stage_jar_')} == {
        'stage_jar_hadoop-bam-7-10-0-jar',
        'stage_jar_htsjdk-2-13-2-jar',
    }


@pytest.mark.parametrize('dag_dir', STAGING_DAG_DIRS, ids=lambda p: p.name)
def test_delta_jars_match_the_delta_spark_package_of_pps(dag_dir: Path) -> None:
    """The python API of delta-spark forwards its calls to the delta jars, so their minor versions must match.

    The jars may be a later patch version, built for the spark of the dataproc images.
    """
    lock = (PPS_CONFIG_PATH.parent / 'uv.lock').read_text()
    package = re.search(r'name = "delta-spark"\nversion = "(\d+\.\d+)\.', lock)
    assert package, 'delta-spark is missing from the pps lock'

    for cluster in StagingDagConfig(dag_dir).dag.infrastructure.clusters.values():
        for coordinate in cluster.jars.get('delta', []):
            assert coordinate.split(':')[2].startswith(f'{package[1]}.'), (
                f'{coordinate} does not match delta-spark {package[1]} of pps'
            )
