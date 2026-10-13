"""Tests for the staging DAGs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from airflow.models import DagBag
from pydantic import ValidationError

from orchestration.dags.config.staging import DATAPROC_INIT_SCRIPT, PPS_CONFIG_PATH, StagingDagConfig
from orchestration.models.staging_dag import StagingDagConfigModel
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
                }
            )
        )

        assert not model.steps['a'].is_batch
        assert model.steps['c'].is_batch
        assert model.scratchpad == {'workers': '4', 'gwas_catalog_top_hits_path': 'gs://prod'}

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
            ({'steps': {'a': {'google_batch': BATCH_STEP['google_batch']}}}, 'needs `google_batch_index_specs`'),
            ({'steps': {'a': {'cluster': 'missing'}}}, 'unknown cluster'),
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

    def test_uris_and_cluster_config(self, tmp_path: Path) -> None:
        config = StagingDagConfig(write_dag(tmp_path, dag_config()))
        cluster = config.cluster_config('cluster')

        assert config.config_uri('a') == 'gs://staging/test_staging_dag/26.9.0/config/a.yaml'
        assert cluster.metadata == {'PPS_REF': 'pps%40v26.9.0'}
        assert cluster.init_actions_uris == [f'gs://staging/test_staging_dag/26.9.0/bin/{DATAPROC_INIT_SCRIPT}']


@pytest.mark.parametrize('dag_dir', STAGING_DAG_DIRS, ids=lambda p: p.name)
def test_pps_steps_of_staging_dags_exist_in_pps(dag_dir: Path) -> None:
    config = StagingDagConfig(dag_dir)

    for step_name, step in config.steps.items():
        if not step.is_batch:
            assert config.step_config(step_name), f'{step_name} missing from {PPS_CONFIG_PATH}'


def test_batch_step_is_mapped_over_its_index(tmp_path: Path) -> None:
    config = dag_config(
        steps={
            'gwas_catalog_top_hits_ingestion': {'cluster': 'cluster'},
            'susie_finemapping': {**BATCH_STEP, 'depends_on': ['gwas_catalog_top_hits_ingestion']},
        }
    )
    dag = build_staging_dag(write_dag(tmp_path, config))
    tasks = {t.task_id: t for t in dag.tasks}

    assert tasks['susie_finemapping_index'].upstream_task_ids == {'gwas_catalog_top_hits_ingestion'}
    assert tasks['susie_finemapping'].upstream_task_ids == {'susie_finemapping_index'}
    assert tasks['susie_finemapping'].is_mapped


def test_gwas_catalog_top_hits_dag(dag_bag: DagBag) -> None:
    dag = dag_bag.dags['gwas_catalog_top_hits']
    tasks = {t.task_id: t for t in dag.tasks}
    steps = ['gwas_catalog_top_hits_ingestion', 'gwas_catalog_top_hits_ld_clumping', 'gwas_catalog_top_hits_pics']
    cluster = 'otg-tophit-gwascatalog'
    create, delete = f'create_cluster_{cluster}', f'delete_cluster_{cluster}'

    assert tasks[create].upstream_task_ids == {f'upload_entrypoint_{cluster}', f'upload_init_script_{cluster}'}
    for step in steps:
        assert {create, f'upload_config_{step}'} <= tasks[step].upstream_task_ids
        assert delete in tasks[step].downstream_task_ids
    assert steps[0] in tasks[steps[1]].upstream_task_ids
    assert steps[1] in tasks[steps[2]].upstream_task_ids
