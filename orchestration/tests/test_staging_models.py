"""Tests for the staging DAG config schema."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from orchestration.dags.config.staging_pipeline import StagingDagConfig
from orchestration.models.staging_dag import (
    BatchStep,
    StagingDagConfigModel,
    StagingEnvironment,
    StagingEnvironments,
    VmStep,
)

BUCKET = 'gs://data_bucket'
ARGS = 'steps.clumping.definition.args'
"""Path to the arguments of a dataproc step of the valid config."""
RUNNABLE = 'steps.finemapping.definition.job.job.task_group.task_config.runnable_spec'
"""Path to the runnable of the batch step of the valid config."""


def raw_config() -> dict[str, Any]:
    """Return a valid staging DAG config, as written in its `config.yaml`."""
    return {
        'description': 'a dag',
        'env': 'test',
        'environments': {
            'prod': {
                'input': {'source': 'gs://source_inputs', 'gnomad': 'gs://gnomad'},
                'output': f'{BUCKET}/prod/20261026T153052',
                'scratchpad': {'gentropy_version': '3.4.0-rc.3', 'num_workers': '10'},
            },
            'test': {
                'input': {
                    'source': f'{BUCKET}/fixtures/20261026T153000/source_inputs',
                    'gnomad': 'gs://gnomad',
                },
                'output': f'{BUCKET}/test/20261026T153052',
                'scratchpad': {'gentropy_version': '4.0.0-dev.1', 'num_workers': '2'},
            },
        },
        'setup': [{'from': 'https://example.com/v1/cli.py', 'to': '{{output}}/bin/cli.py'}],
        'infrastructure': {'clusters': {'main': {'num_workers': 2}}},
        'steps': {
            'study_index': {
                'kind': 'dataproc',
                'definition': {
                    'cluster': 'main',
                    'main': '{{output}}/bin/cli.py',
                    'args': (
                        'step=gwas_catalog_study_index \\\n'
                        '  "step.catalog_study_files=[\'{{input.source}}/studies/studies.tsv\']" \\\n'
                        '  step.study_index_path={{output}}/output/study_index\n'
                    ),
                },
            },
            'clumping': {
                'depends_on': ['study_index'],
                'kind': 'dataproc',
                'definition': {
                    'cluster': 'main',
                    'main': '{{output}}/bin/cli.py',
                    'args': (
                        'step=clumping \\\n'
                        '  step.summary_statistics_input_path={{input.source}}/sumstats/*/*.parquet \\\n'
                        '  step.ld_index_path={{input.gnomad}}/ld_index \\\n'
                        '  step.output_path={{output}}/output/clumped \\\n'
                        '  "+step.session.extended_spark_conf={spark.x: \'1\'}"\n'
                    ),
                },
            },
            'finemapping': {
                'depends_on': ['clumping'],
                'kind': 'batch',
                'definition': {
                    'index': {
                        'pointer': 'finemapping',
                        'max_task_count': 100,
                        'generator_specs': {'generator_options': {'output_path': '{{output}}/output/susie'}},
                    },
                    'job': {
                        'job': {
                            'task_group': {
                                'parallelism': 10,
                                'task_environments': {'environments': []},
                                'task_config': {
                                    'instance_resource_spec': {
                                        'cpu_milli': 4000,
                                        'memory_mib': 25000,
                                        'boot_disk_mib': 20000,
                                    },
                                    'runnable_spec': {
                                        'image_uri': 'gentropy:4.0.0-dev.1',
                                        'inline_commands': [
                                            '-c',
                                            (
                                                'gentropy step=susie step.locus=$INPUT_PARTITION '
                                                'step.out={{output}}/output/susie'
                                            ),
                                        ],
                                    },
                                },
                            },
                            'allocation': {'instance': {'machine_type': 'n2-highmem-4'}},
                            'logs': {},
                        },
                    },
                },
            },
            'annotation': {
                'depends_on': ['study_index'],
                'kind': 'vm',
                'definition': {
                    'image': 'gentropy:4.0.0-dev.1',
                    'machine_type': 'n2-highmem-16',
                    'command': 'gentropy step=annotation step.variants={{input.gnomad}}/v',
                },
            },
        },
    }


def config(env: str = 'test') -> dict[str, Any]:
    """Return the valid config rendered for an environment, as `StagingDagConfig` renders it."""
    raw = raw_config()
    raw['env'] = env
    text = yaml.safe_dump(raw)
    for name, value in StagingEnvironment.model_validate(raw['environments'][env]).template_context.items():
        text = text.replace(f'{{{{{name}}}}}', value)
    return yaml.safe_load(text)


OUTPUT = f'{BUCKET}/test/20261026T153052'
"""The run folder of the rendered test config."""


def with_change(path: str, value: Any, c: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return the valid rendered config, or `c`, with the value at a dotted path replaced."""
    c = copy.deepcopy(config() if c is None else c)
    *parents, key = path.split('.')
    node = c
    for parent in parents:
        node = node[int(parent)] if isinstance(node, list) else node[parent]
    if isinstance(node, list):
        node[int(key)] = value
    else:
        node[key] = value
    return c


def validate(c: dict[str, Any]) -> tuple[StagingEnvironments, StagingDagConfigModel]:
    """Validate a rendered config as `StagingDagConfig` does: the environments, then the rest."""
    environments = StagingEnvironments.model_validate(c)
    rest = {k: v for k, v in c.items() if k not in StagingEnvironments.model_fields}
    return environments, StagingDagConfigModel.model_validate(rest)


class TestStagingDagConfigModel:
    """Validation of the rendered staging DAG config."""

    def test_valid_config(self) -> None:
        """A config following every rule validates."""
        environments, model = validate(config())
        assert environments.environment.template_context['gentropy_version'] == '4.0.0-dev.1'
        assert environments.environment.config_root == f'{BUCKET}/test/20261026T153052/config'
        assert {name: step.kind for name, step in model.steps.items()} == {
            'study_index': 'dataproc',
            'clumping': 'dataproc',
            'finemapping': 'batch',
            'annotation': 'vm',
        }

    def test_test_input_roots(self) -> None:
        """A test input root is the prod root, or a dated static slice of it in the data bucket."""
        environments, _ = validate(config())
        assert environments.environment.input == {
            'source': f'{BUCKET}/fixtures/20261026T153000/source_inputs',
            'gnomad': 'gs://gnomad',
        }
        assert environments.environment.fixtures_root == f'{BUCKET}/fixtures'

    def test_prod_reads_its_cache(self) -> None:
        """A prod input root can be in the data bucket, such as the cache."""
        c = config('prod')
        c['environments']['prod']['input']['source'] = f'{BUCKET}/cache'
        c['environments']['test']['input']['source'] = f'{BUCKET}/cache'
        environments, _ = validate(c)
        assert environments.environment.input['source'] == f'{BUCKET}/cache'

    def test_dataproc_steps_need_the_infrastructure(self) -> None:
        """Without the infrastructure block, the config contains no cluster, so a dataproc step is invalid."""
        c = config()
        del c['infrastructure']
        with pytest.raises(
            ValidationError, match=r"clusters \['main'\] of step \w+ are not in infrastructure.clusters \[\]"
        ):
            validate(c)

    def test_config_without_infrastructure(self) -> None:
        """Without dataproc steps, the infrastructure block is optional."""
        c = config()
        del c['infrastructure']
        del c['steps']['study_index'], c['steps']['clumping']
        c['steps']['finemapping']['depends_on'] = []
        c['steps']['annotation']['depends_on'] = []
        assert validate(c)[1].infrastructure.clusters == {}

    def test_config_without_test_environment(self) -> None:
        """The test environment is optional."""
        c = config('prod')
        del c['environments']['test']
        environments, _ = validate(c)
        assert list(environments.environments) == ['prod']

    @pytest.mark.parametrize(
        ('change', 'error'),
        [
            pytest.param(lambda c: c['environments'].pop('test'), 'test is not defined', id='selected-test-missing'),
            pytest.param(
                lambda c: (c.update(env='test'), c['environments'].pop('prod')),
                'prod environment is required',
                id='prod-missing',
            ),
        ],
    )
    def test_invalid_environments(self, change: Any, error: str) -> None:
        """The prod environment is required, and so is the selected environment."""
        c = config()
        change(c)
        with pytest.raises(ValidationError, match=error):
            validate(c)

    @pytest.mark.parametrize(
        ('path', 'value', 'error'),
        [
            pytest.param('environments.prod.scratchpad', {'output': 'x'}, 'reserved', id='reserved-scratchpad'),
            pytest.param(
                'environments.test.output', f'{BUCKET}/test/output', 'is not <data bucket>', id='undated-output'
            ),
            pytest.param(
                'environments.test.output',
                f'{BUCKET}/prod/20261026T153052',
                'is not <data bucket>',
                id='test-output-in-prod',
            ),
            pytest.param(
                'environments.test.output',
                'gs://other/test/20261026T153052',
                'data buckets differ',
                id='other-bucket',
            ),
            pytest.param(
                'environments.test.input.source',
                f'{BUCKET}/test/input/source',
                'it can only be',
                id='test-input-elsewhere',
            ),
            pytest.param(
                'environments.test.input.source',
                f'{BUCKET}/test/20261026T153052/input/source',
                'it can only be',
                id='test-input-copy',
            ),
            pytest.param('environments.test.input', {'source': 'x'}, 'input names differ', id='input-names'),
            pytest.param(
                'environments.test.input.source',
                f'{BUCKET}/fixtures',
                'it can only be',
                id='fixtures-without-name',
            ),
            pytest.param(
                'environments.test.input.source',
                f'{BUCKET}/fixtures/source_inputs',
                'it can only be',
                id='undated-fixtures',
            ),
            pytest.param(
                'environments.test.input.source',
                f'{BUCKET}/fixtures/20261026/source_inputs',
                'it can only be',
                id='fixtures-date-only',
            ),
            pytest.param(
                'environments.test.input.source',
                f'{BUCKET}/fixtures/20261026T153000/source',
                'it can only be',
                id='fixtures-by-root-name',
            ),
            pytest.param(ARGS, "step='c", 'not valid shell syntax', id='unclosed-quote'),
            pytest.param('steps.clumping.definition.cluster', 'other', 'not in infrastructure.clusters', id='cluster'),
            pytest.param('infrastructure.clusters.spare', {}, 'not the cluster of any step', id='unused-cluster'),
            pytest.param('infrastructure.clusters.main.num_worker', 2, 'Extra inputs', id='cluster-typo'),
            pytest.param(
                'infrastructure.clusters.main.project_id',
                'open-targets-eu-dev',
                'project of a staging cluster',
                id='cluster-project',
            ),
            pytest.param('setup.0.from', 'gs://bucket/cli.py', 'http', id='setup-from'),
            pytest.param(
                'setup',
                [
                    {'from': 'https://a/cli.py', 'to': f'{OUTPUT}/bin/cli.py'},
                    {'from': 'https://b/cli.py', 'to': f'{OUTPUT}/bin/cli.py'},
                ],
                'more than once',
                id='setup-twice',
            ),
            pytest.param(
                'steps.clumping.kind', 'spark', "Input should be 'dataproc', 'batch' or 'vm'", id='unknown-kind'
            ),
            pytest.param('steps.clumping.kind', 'vm', 'image', id='fields-of-another-kind'),
            pytest.param('steps.clumping.definition', None, 'valid dictionary', id='no-definition'),
            pytest.param('steps.clumping.definition.kind', 'dataproc', 'Extra inputs', id='kind-in-definition'),
            pytest.param(RUNNABLE, {'inline_commands': ['-c', 'x']}, 'image_uri', id='runnable-without-image'),
            pytest.param(
                'steps.finemapping.definition.index.max_tasks_count', 99, 'Extra inputs', id='batch-index-typo'
            ),
            pytest.param(f'{RUNNABLE}.entrypoints', '/bin/sh', 'Extra inputs', id='batch-runnable-typo'),
            pytest.param(
                'steps.finemapping.definition.job.job.labels',
                {'team': 'x'},
                'labels of the step',
                id='batch-job-labels',
            ),
            pytest.param(
                'steps.finemapping.definition.job.job.allocation.labels',
                {'team': 'x'},
                'labels of the step',
                id='batch-allocation-labels',
            ),
            pytest.param('steps.clumping.depends_on', ['nope'], 'unknown steps', id='dependency'),
            pytest.param('steps.study_index.depends_on', ['clumping'], 'cycle', id='cycle'),
        ],
    )
    def test_invalid_config(self, path: str, value: Any, error: str) -> None:
        """A config breaking a rule is invalid, with the rule in the error."""
        with pytest.raises(ValidationError, match=error):
            validate(with_change(path, value))


class TestStagingDagConfig:
    """Reading a staging DAG config with the values of its environment."""

    def test_renders_the_selected_environment(self, tmp_path: Path) -> None:
        """The environment placeholders are rendered with the values of the selected environment."""
        raw = raw_config()
        raw['infrastructure']['clusters']['main']['num_workers'] = '{{num_workers}}'
        raw['steps']['annotation']['definition']['command'] = (
            'gentropy step=annotation step.version={{gentropy_version}}'
        )
        dag_dir = tmp_path / 'a_dag'
        dag_dir.mkdir()
        (dag_dir / 'config.yaml').write_text(yaml.safe_dump(raw))

        for env, num_workers, version in [('test', 2, '4.0.0-dev.1'), ('prod', 10, '3.4.0-rc.3')]:
            raw['env'] = env
            (dag_dir / 'config.yaml').write_text(yaml.safe_dump(raw))
            dag = StagingDagConfig(dag_dir).dag
            assert dag.infrastructure.clusters['main'].num_workers == num_workers
            vm = dag.steps['annotation'].definition
            assert isinstance(vm, VmStep)
            assert vm.command == f'gentropy step=annotation step.version={version}'

    @pytest.mark.parametrize(
        ('path', 'value', 'error'),
        [
            pytest.param(ARGS, 'step=c step.x={{nope}}/x', r'unrendered placeholders: \{\{nope\}\}', id='step'),
            pytest.param(
                'infrastructure.clusters.main.num_workers', '{{nope}}', 'unrendered placeholders', id='cluster'
            ),
            pytest.param(ARGS, 'step=c step.n={# 1 #}', r"Jinja tags \['\{#'\]", id='jinja-comment'),
            pytest.param(ARGS, 'step=c {% if x %}', r"Jinja tags \['\{%'\]", id='jinja-statement'),
            pytest.param(
                'environments.test.input.source',
                '{{output}}/input/source',
                r'unrendered placeholders: \{\{output\}\}',
                id='environment',
            ),
        ],
    )
    def test_rejects_unrendered_text(self, tmp_path: Path, path: str, value: str, error: str) -> None:
        """The rendered text of the config, and the raw environments, contain no placeholder."""
        dag_dir = tmp_path / 'a_dag'
        dag_dir.mkdir()
        (dag_dir / 'config.yaml').write_text(yaml.safe_dump(with_change(path, value, raw_config())))
        with pytest.raises(ValueError, match=error):
            StagingDagConfig(dag_dir)

    def test_rejects_unrendered_placeholders_in_comments(self, tmp_path: Path) -> None:
        """The record of a run is the rendered text of the config, so its comments are checked too."""
        dag_dir = tmp_path / 'a_dag'
        dag_dir.mkdir()
        (dag_dir / 'config.yaml').write_text('# reads {{nope}}\n' + yaml.safe_dump(raw_config()))
        with pytest.raises(ValueError, match=r'unrendered placeholders: \{\{nope\}\}'):
            StagingDagConfig(dag_dir)

    def test_rejects_jinja_tags_in_comments(self, tmp_path: Path) -> None:
        """A Jinja tag in a comment breaks the template of the record of the run."""
        dag_dir = tmp_path / 'a_dag'
        dag_dir.mkdir()
        (dag_dir / 'config.yaml').write_text('# the {# 1 #} step\n' + yaml.safe_dump(raw_config()))
        with pytest.raises(ValueError, match='Jinja tags'):
            StagingDagConfig(dag_dir)

    def test_rejects_an_undefined_environment(self, tmp_path: Path) -> None:
        """The selected environment is required."""
        raw = raw_config()
        del raw['environments']['test']
        dag_dir = tmp_path / 'a_dag'
        dag_dir.mkdir()
        (dag_dir / 'config.yaml').write_text(yaml.safe_dump(raw))
        with pytest.raises(ValidationError, match='environment test is not defined'):
            StagingDagConfig(dag_dir)

    def test_renders_every_kind_of_step(self, tmp_path: Path) -> None:
        """The whole config is rendered, for every kind of step."""
        dag_dir = tmp_path / 'a_dag'
        dag_dir.mkdir()
        (dag_dir / 'config.yaml').write_text(yaml.safe_dump(raw_config()))
        dag = StagingDagConfig(dag_dir).dag
        batch, vm = dag.steps['finemapping'].definition, dag.steps['annotation'].definition
        assert isinstance(batch, BatchStep)
        assert isinstance(vm, VmStep)
        assert batch.index.generator_specs.generator_options == {'output_path': f'{OUTPUT}/output/susie'}
        assert batch.job.job.task_group.task_config.runnable_spec.inline_commands == [
            '-c',
            f'gentropy step=susie step.locus=$INPUT_PARTITION step.out={OUTPUT}/output/susie',
        ]
        assert vm.command == 'gentropy step=annotation step.variants=gs://gnomad/v'
