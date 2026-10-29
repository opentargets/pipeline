r"""Steps of the staging DAGs, executed on Dataproc, on Google Batch or on a VM.

The `kind` of a step selects the model of its `definition`: `DataprocStep`, `BatchStep` or
`VmStep`, and `build()` maps the definition to the operator of the step. The program of
a step is part of the definition, with no assumption on the program:

```yaml
pics:
  kind: dataproc
  depends_on: [ld_based_clumping]
  definition:
    cluster: otg-tophit-gwascatalog
    main: gs://gwas_catalog_top_hits/prod/20261026T153052/bin/cli.py
    args: |
      step=pics \
      step.study_locus_ld_annotated_in=gs://gwas_catalog_top_hits/prod/20261026T153052/output/study_locus_ld_clumped \
      step.session.write_mode=overwrite
```
"""

from __future__ import annotations

import shlex
from enum import StrEnum
from typing import Any, Self

from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.sdk import TaskGroup
from airflow.task.trigger_rule import TriggerRule
from google.cloud.dataproc_v1.types.jobs import PySparkJob
from pydantic import BaseModel, ConfigDict, ValidationInfo, field_validator, model_validator

from orchestration.models.batch.operator import BatchIndexOperatorSpec, BatchJobOperatorSpec
from orchestration.operators.batch import BatchCollectOperator, BatchIndexOperator, BatchJobOperator
from orchestration.operators.dataproc import SubmitJobOperator
from orchestration.operators.gce import ComputeEngineRunContainerizedWorkloadSensor, DeleteInstanceOperator
from orchestration.utils.common import GCP_PROJECT_GENETICS, GCP_REGION
from orchestration.utils.labels import Labels
from orchestration.utils.staging import run_resource_name
from orchestration.utils.utils import clean_name


class StepKind(StrEnum):
    """The kind of runtime of a step, which selects the model of its definition."""

    DATAPROC = 'dataproc'
    BATCH = 'batch'
    VM = 'vm'


def split_args(args: str) -> list[str]:
    r"""Return the arguments of a string written with shell syntax.

    `\` at the end of a line continues the line, and quotes group an argument with spaces.

    Examples:
        >>> split_args('''step=pics \\
        ...   step.picsed_study_locus_out=gs://gwas_catalog_top_hits/prod/20261026T153052/output/credible_set \\
        ...   "step.catalog_study_files=['gs://gwas_catalog_inputs/studies.tsv']"
        ... ''')
        ['step=pics', 'step.picsed_study_locus_out=gs://gwas_catalog_top_hits/prod/20261026T153052/output/credible_set', "step.catalog_study_files=['gs://gwas_catalog_inputs/studies.tsv']"]

    Raises:
        ValueError: If the string is not valid shell syntax.
    """  # noqa: E501
    try:
        return shlex.split(args.replace('\\\n', ' '))
    except ValueError as e:
        raise ValueError(f'the arguments are not valid shell syntax: {e}') from e


class StepDefinition(BaseModel):
    """The definition of a step, for one kind of runtime.

    `build()` maps the definition to the tasks of the step, in a task group named after the
    step, with the labels of the step on every resource of the step. The tasks downstream of
    the step depend on the whole group.
    `clusters` are the names of the infrastructure clusters of the definition, empty for a
    runtime without infrastructure clusters.
    """

    model_config = ConfigDict(extra='forbid')

    @property
    def clusters(self) -> frozenset[str]:
        """The names of the infrastructure clusters of the definition."""
        return frozenset()

    def build(self, task_id: str, labels: Labels) -> TaskGroup:
        """Return the task group of the step, created in the current DAG context.

        Args:
            task_id: The id of the task group, the name of the step.
            labels: The labels of the resources of the step.

        Raises:
            NotImplementedError: If the definition of a kind of runtime does not implement it.
        """
        raise NotImplementedError(f'step {task_id}: the tasks of {type(self).__name__} are not implemented')


class DataprocStep(StepDefinition):
    r"""A PySpark job on a Dataproc cluster of the DAG: the python file `main`, with `args`.

    `main` is usually a file copied into the `bin/` folder of the run by `setup`. `args`
    are the arguments of `main`, passed as written: a list, or a string with shell syntax.

    Examples:
        >>> from pydantic import ValidationError
        >>> step = DataprocStep.model_validate({
        ...     'cluster': 'otg-tophit-gwascatalog',
        ...     'main': 'gs://gwas_catalog_top_hits/prod/20261026T153052/bin/cli.py',
        ...     'args': '''step=pics \\
        ...       step.study_locus_ld_annotated_in=gs://gwas_catalog_top_hits/prod/20261026T153052/output/study_locus_ld_clumped \\
        ...       step.session.write_mode=overwrite
        ...     ''',
        ... })
        >>> step.args
        ['step=pics', 'step.study_locus_ld_annotated_in=gs://gwas_catalog_top_hits/prod/20261026T153052/output/study_locus_ld_clumped', 'step.session.write_mode=overwrite']

        A list is passed as it is:

        >>> DataprocStep(cluster='c', main='gs://b/main.py', args=['--input', 'gs://b/in']).args
        ['--input', 'gs://b/in']

        A string is valid shell syntax:

        >>> try:
        ...     DataprocStep(cluster='c', main='gs://b/main.py', args="--name 'pics")
        ... except ValidationError as e:
        ...     print(e.errors()[0]['msg'])
        Value error, the arguments are not valid shell syntax: No closing quotation
    """  # noqa: E501

    cluster: str
    """The cluster of the step, from `infrastructure.clusters`."""
    main: str
    """The python file of the job, usually a file of the `bin/` folder of the run."""
    args: list[str] = []
    """The arguments of `main`: a list, or a string with shell syntax."""

    @field_validator('args', mode='before')
    @classmethod
    def _split_args(cls, args: Any) -> Any:
        return split_args(args) if isinstance(args, str) else args

    @property
    def clusters(self) -> frozenset[str]:
        """The cluster of the step."""
        return frozenset({self.cluster})

    def build(self, task_id: str, labels: Labels) -> TaskGroup:
        """Return the task group of the step: `run`, the PySpark job submitted to the cluster of the run."""
        with TaskGroup(group_id=task_id) as group:
            SubmitJobOperator(
                task_id='run',
                project_id=GCP_PROJECT_GENETICS,
                region=GCP_REGION,
                cluster_name=run_resource_name(self.cluster),
                step_name=task_id,
                py_spark_job=PySparkJob(main_python_file_uri=self.main, args=self.args),
                labels=labels,
                deferrable=True,
            )
        return group


class VmStep(StepDefinition):
    """A shell command executed in a container image, on a VM dedicated to the step.

    As for the PTS steps, the VM is created before the step and deleted after it, in the
    genetics project, and the VM settings are common to every VM step. `machine_type` overrides the default machine
    type for one step.

    Examples:
        >>> step = VmStep.model_validate({
        ...     'image': 'europe-west1-docker.pkg.dev/open-targets-genetics-dev/opentargets/gentropy:4.0.0-dev.1',
        ...     'machine_type': 'n2d-highmem-32',
        ...     'command': 'gentropy step=variant_index step.output_path=gs://gwas_catalog_top_hits/prod/20261026T153052/output/variant_index',
        ... })
        >>> step.machine_type
        'n2d-highmem-32'
    """

    image: str
    """The container image of the command."""
    command: str
    """The shell command of the step, executed as written."""
    machine_type: str = 'n1-standard-16'
    """Machine type of the VM. The default is the default of the PTS VMs."""

    def build(self, task_id: str, labels: Labels) -> TaskGroup:
        """Return the task group of the step: `run`, `delete_vm` and `done`.

        `run` creates the VM `staging-<step>-<hash>` (`run_resource_name`), and runs the command
        in the image.
        `delete_vm` follows `run` whatever its result, and `done` follows `run` only, so a
        failed run is a failed step, with the VM deleted.
        """
        instance_name = run_resource_name(f'staging-{task_id}')
        with TaskGroup(group_id=task_id) as group:
            run = ComputeEngineRunContainerizedWorkloadSensor(
                task_id='run',
                project=GCP_PROJECT_GENETICS,
                instance_name=instance_name,
                labels=labels,
                container_image=self.image,
                container_command=self.command.replace('\\\n', ' ').strip(),
                deferrable=True,
                machine_type=self.machine_type,
            )
            delete = DeleteInstanceOperator(
                task_id='delete_vm',
                project_id=GCP_PROJECT_GENETICS,
                resource_id=instance_name,
                trigger_rule=TriggerRule.ALL_DONE,
            )
            run >> [delete, EmptyOperator(task_id='done')]
        return group


class BatchStep(StepDefinition):
    """A Google Batch job, fanned out over an index: the specs of the Batch DAGs, as they are.

    `index` is a `BatchIndexOperatorSpec` and `job` a `BatchJobOperatorSpec`. The runnable
    of the job (`runnable_spec`) is the image and the command of each task. The variables
    of the index rows, such as `$INPUT_PARTITION`, are available in the command.

    Examples:
        >>> step = BatchStep.model_validate({
        ...     'index': {
        ...         'pointer': 'finemapping',
        ...         'max_task_count': 40000,
        ...         'generator_specs': {'generator_options': {'output_path': 'gs://gwas_catalog_top_hits/prod/20261026T153052/output/credible_set'}},
        ...     },
        ...     'job': {
        ...         'job': {
        ...             'task_group': {
        ...                 'parallelism': 10000,
        ...                 'task_environments': {'environments': []},
        ...                 'task_config': {
        ...                     'instance_resource_spec': {'cpu_milli': 4000, 'memory_mib': 25000, 'boot_disk_mib': 20000},
        ...                     'runnable_spec': {
        ...                         'image_uri': 'europe-west1-docker.pkg.dev/open-targets-genetics-dev/opentargets/gentropy:4.0.0-dev.1',
        ...                         'inline_commands': ['-c', 'gentropy step=susie_finemapping step.study_locus_manifest_path=$INPUT_PARTITION'],
        ...                     },
        ...                 },
        ...             },
        ...             'allocation': {'instance': {'machine_type': 'n2-highmem-4', 'provisioning_model': 'SPOT'}},
        ...             'logs': {},
        ...         },
        ...     },
        ... })
        >>> runnable = step.job.job.task_group.task_config.runnable_spec
        >>> runnable.entrypoint, runnable.inline_commands
        ('/bin/bash', ['-c', 'gentropy step=susie_finemapping step.study_locus_manifest_path=$INPUT_PARTITION'])
    """  # noqa: E501

    index: BatchIndexOperatorSpec
    """The index the job is fanned out over."""
    job: BatchJobOperatorSpec
    """The job, run once per index row. Its labels are the labels of the step."""

    @model_validator(mode='after')
    def _validate_labels(self) -> Self:
        """Check that the job spec contains no labels: the labels of the step replace them."""
        if self.job.job.labels or self.job.job.allocation.labels:
            raise ValueError('the labels of a batch job are the labels of the step, not labels of the job spec')
        return self

    def build(self, task_id: str, labels: Labels) -> TaskGroup:
        """Return the task group of the step: `index`, then `job` per index row, then `collect`.

        `collect` is part of the group only with a collect spec in `job`.
        """
        with TaskGroup(group_id=task_id) as group:
            index = BatchIndexOperator(task_id='index', batch_index_specs=self.index)
            jobs = BatchJobOperator.partial(
                task_id='job',
                job_name=clean_name(task_id),
                batch_job_spec=self.job,
                labels=labels,
            ).expand(batch_index_row=index.output)
            if self.job.collect is not None:
                jobs >> BatchCollectOperator(task_id='collect', collect_spec=self.job.collect)
        return group


class StagingStep(BaseModel):
    """A step of a staging DAG: its kind of runtime, its dependencies, and its definition.

    The `kind` of the step selects the model of the `definition`, so the validation errors
    of a definition are the errors of that model only.

    Examples:
        >>> from pydantic import ValidationError
        >>> step = StagingStep.model_validate({
        ...     'kind': 'dataproc',
        ...     'depends_on': ['ld_based_clumping'],
        ...     'definition': {
        ...         'cluster': 'otg-tophit-gwascatalog',
        ...         'main': 'gs://gwas_catalog_top_hits/prod/20261026T153052/bin/cli.py',
        ...         'args': 'step=pics',
        ...     },
        ... })
        >>> type(step.definition).__name__, step.definition.clusters, step.depends_on
        ('DataprocStep', frozenset({'otg-tophit-gwascatalog'}), ['ld_based_clumping'])
        >>> from airflow.sdk import DAG
        >>> with DAG(dag_id='a_dag', schedule=None):
        ...     group = step.definition.build('pics', Labels(extra={'step': 'pics'}))
        >>> group.group_id, list(group.children)
        ('pics', ['pics.run'])

        The errors of a definition are the errors of the model of its kind:

        >>> try:
        ...     StagingStep.model_validate({'kind': 'dataproc', 'definition': {'clustr': 'c', 'main': 'gs://b/cli.py'}})
        ... except ValidationError as e:
        ...     for error in e.errors():
        ...         print('.'.join(map(str, error['loc'])), '|', error['msg'])
        definition.cluster | Field required
        definition.clustr | Extra inputs are not permitted

        The `kind` of a step is one of `dataproc`, `batch` or `vm`:

        >>> try:
        ...     StagingStep.model_validate({'kind': 'spark', 'definition': {'command': 'spark-submit job.py'}})
        ... except ValidationError as e:
        ...     print(e.errors()[0]['msg'])
        Input should be 'dataproc', 'batch' or 'vm'
    """

    model_config = ConfigDict(extra='forbid')

    kind: StepKind
    """The kind of runtime of the step, which selects the model of the definition."""
    depends_on: list[str] = []
    """Steps that must finish before this one starts."""
    definition: DataprocStep | BatchStep | VmStep
    """The definition of the step, for its kind of runtime."""

    @field_validator('definition', mode='before')
    @classmethod
    def _dispatch_definition(cls, definition: Any, info: ValidationInfo) -> Any:
        """Validate the definition with the model of the `kind` of the step."""
        kind: StepKind | None = info.data.get('kind')
        match kind:
            case StepKind.DATAPROC:
                return DataprocStep.model_validate(definition)
            case StepKind.BATCH:
                return BatchStep.model_validate(definition)
            case StepKind.VM:
                return VmStep.model_validate(definition)
            case None:
                return definition  # the invalid kind is reported on its own field
