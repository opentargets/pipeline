"""The config of a staging DAG, and its rules."""

from __future__ import annotations

import re
from graphlib import CycleError, TopologicalSorter
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from orchestration.models.staging_dag.infrastructure import StagingInfrastructure
from orchestration.models.staging_dag.setup import SetupCopy
from orchestration.models.staging_dag.step import StagingStep

MAX_STEP_NAME_LENGTH = 38
"""Maximum length of a step name, for the names of the resources of the step, at most 63 characters.

The tightest is the name of a Batch job, `<step>-job-<index>-<YYYYmmdd-HHMMSS>`, with up to
9999 jobs per step. A VM is `staging-<step>-<hash>`, and a Dataproc job id
`<cluster>-<hash>-<step>-<id>`, at most 100 characters.
"""


class StagingDagConfigModel(BaseModel):
    r"""The `config.yaml` of a staging DAG without its environments, rendered with the values of one of them.

    The config is read with `StagingDagConfig`: the environments are validated once, on the raw
    config, with `StagingEnvironments`, the config is rendered with the values of the
    selected environment, and the rest of the config is validated with this model. A
    rendered config contains no placeholder.

    Examples:
        The GWAS Catalog top hits DAG, rendered for prod:

        >>> import yaml
        >>> from pydantic import ValidationError
        >>> config = StagingDagConfigModel.model_validate(yaml.safe_load('''
        ... description: GWAS Catalog curated associations → study index and PICS credible sets
        ...
        ... setup:
        ...   - from: https://raw.githubusercontent.com/opentargets/gentropy/v4.0.0-dev.1/utils/install_dependencies_on_cluster.sh
        ...     to: gs://gwas_catalog_top_hits/prod/20261026T153052/bin/install_dependencies_on_cluster.sh
        ...   - from: https://raw.githubusercontent.com/opentargets/gentropy/v4.0.0-dev.1/src/gentropy/cli.py
        ...     to: gs://gwas_catalog_top_hits/prod/20261026T153052/bin/cli.py
        ...
        ... infrastructure:
        ...   clusters:
        ...     otg-tophit-gwascatalog:
        ...       project_id: open-targets-genetics-dev
        ...       autoscaling_policy: otg-gwascatalog-tophit
        ...       metadata:
        ...         GENTROPY_REF: v4.0.0-dev.1
        ...       init_actions_uris:
        ...         - gs://gwas_catalog_top_hits/prod/20261026T153052/bin/install_dependencies_on_cluster.sh
        ...
        ... steps:
        ...   gwas_catalog_top_hit_ingestion:
        ...     kind: dataproc
        ...     definition:
        ...       cluster: otg-tophit-gwascatalog
        ...       main: gs://gwas_catalog_top_hits/prod/20261026T153052/bin/cli.py
        ...       args: |
        ...         step=gwas_catalog_top_hit_ingestion \\
        ...           "step.catalog_study_files=['gs://gwas_catalog_inputs/gentroutils/20260802/gwas_catalog_download_studies.tsv']" \\
        ...           step.variant_annotation_path=gs://gnomad_data_2/v4.1/variant_index \\
        ...           step.catalog_studies_out=gs://gwas_catalog_top_hits/prod/20261026T153052/output/study_index \\
        ...           step.session.spark_uri=yarn
        ...   pics:
        ...     depends_on: [gwas_catalog_top_hit_ingestion]
        ...     kind: dataproc
        ...     definition:
        ...       cluster: otg-tophit-gwascatalog
        ...       main: gs://gwas_catalog_top_hits/prod/20261026T153052/bin/cli.py
        ...       args: |
        ...         step=pics \\
        ...           step.study_index_path=gs://gwas_catalog_top_hits/prod/20261026T153052/output/study_index \\
        ...           step.picsed_study_locus_out=gs://gwas_catalog_top_hits/prod/20261026T153052/output/credible_set \\
        ...           step.session.spark_uri=yarn
        ... '''))

        The steps, with the runtime and the dependencies of each:

        >>> {name: (step.kind, step.depends_on) for name, step in config.steps.items()}
        {'gwas_catalog_top_hit_ingestion': (<StepKind.DATAPROC: 'dataproc'>, []), 'pics': (<StepKind.DATAPROC: 'dataproc'>, ['gwas_catalog_top_hit_ingestion'])}

        The setup is optional:

        >>> no_setup = config.model_dump(by_alias=True, exclude_unset=True)
        >>> del no_setup['setup']
        >>> StagingDagConfigModel.model_validate(no_setup).setup
        []

        A broken rule is a validation error:

        >>> def check(change):
        ...     raw = config.model_dump(by_alias=True, exclude_unset=True)
        ...     change(raw)
        ...     try:
        ...         StagingDagConfigModel.model_validate(raw)
        ...     except ValidationError as e:
        ...         print(e.errors()[0]['msg'])

        The cluster of a Dataproc step is a cluster of the infrastructure:

        >>> check(lambda c: c['steps']['pics']['definition'].update(cluster='otg-other'))
        Value error, clusters ['otg-other'] of step pics are not in infrastructure.clusters ['otg-tophit-gwascatalog']

        Including when the config contains no infrastructure:

        >>> check(lambda c: c.pop('infrastructure'))
        Value error, clusters ['otg-tophit-gwascatalog'] of step gwas_catalog_top_hit_ingestion are not in infrastructure.clusters []

        The environments are validated by `StagingEnvironments`, and are not fields of this model:

        >>> check(lambda c: c.update(env='prod'))
        Extra inputs are not permitted

        A step name is the start of the names of the resources of the step:

        >>> check(lambda c: c['steps'].update(PICS=c['steps'].pop('pics')))
        Value error, step name 'PICS' is not lowercase letters, digits, underscores and dashes, starting with a letter
        >>> check(lambda c: c['steps'].update(pics_with_a_name_longer_than_the_batch_limit=c['steps'].pop('pics')))
        Value error, step name 'pics_with_a_name_longer_than_the_batch_limit' is longer than 38 characters, the maximum for the names of its resources
    """  # noqa: E501

    model_config = ConfigDict(extra='forbid')

    description: str
    """Description of the DAG."""
    setup: list[SetupCopy] = []
    """Files copied into the `bin/` folder of the run before the steps. Optional: required
    only when a cluster or a step refers to a file of `bin/`."""
    infrastructure: StagingInfrastructure = Field(default_factory=StagingInfrastructure)
    """The Dataproc clusters of the steps. Optional: without it, the config contains no
    cluster, so the config is valid only without Dataproc steps."""
    steps: dict[str, StagingStep]

    @model_validator(mode='after')
    def _validate_step_names(self) -> Self:
        """Check that each step name is valid at the start of the names of the resources of the step."""
        for name in self.steps:
            if not re.fullmatch(r'[a-z][a-z0-9_-]*', name):
                raise ValueError(
                    f'step name {name!r} is not lowercase letters, digits, underscores and dashes, '
                    'starting with a letter'
                )
            if len(name) > MAX_STEP_NAME_LENGTH:
                raise ValueError(
                    f'step name {name!r} is longer than {MAX_STEP_NAME_LENGTH} characters, '
                    'the maximum for the names of its resources'
                )
        return self

    @model_validator(mode='after')
    def _validate_steps(self) -> Self:
        """Check the references between the steps, and between the steps and the clusters.

        * the clusters of each step definition are in `infrastructure.clusters`, so a typo in
          a cluster name is a validation error, and not a job submission to an absent cluster,
        * each `depends_on` entry is a step of the DAG. This check precedes the cycle check:
          an unknown name would be an extra step without dependencies in the sort,
        * the dependencies are acyclic. `static_order()` is lazy, so `tuple()` forces the
          sort, and `CycleError` names the steps of the cycle,
        * each cluster is the cluster of at least one step. The delete task of a cluster
          follows the last step of the cluster, so a cluster without steps is deleted only by
          its idle TTL.

        The clusters of a step are `definition.clusters`, empty for a runtime without
        infrastructure clusters, such as Batch or a VM.
        """
        for name, step in self.steps.items():
            if missing := sorted(step.definition.clusters - set(self.infrastructure.clusters)):
                raise ValueError(
                    f'clusters {missing} of step {name} are not in '
                    f'infrastructure.clusters {sorted(self.infrastructure.clusters)}'
                )
            if unknown := set(step.depends_on) - set(self.steps):
                raise ValueError(f'step {name} depends on unknown steps {sorted(unknown)}')
        try:
            tuple(TopologicalSorter({n: s.depends_on for n, s in self.steps.items()}).static_order())
        except CycleError as e:
            raise ValueError(f'steps form a cycle: {e.args[1]}') from e

        used = {cluster for step in self.steps.values() for cluster in step.definition.clusters}
        if unused := set(self.infrastructure.clusters) - used:
            raise ValueError(f'clusters {sorted(unused)} are not the cluster of any step')
        return self

    @model_validator(mode='after')
    def _validate_setup(self) -> Self:
        copied = [copy.file_name for copy in self.setup]
        if duplicated := sorted({name for name in copied if copied.count(name) > 1}):
            raise ValueError(f'setup files {duplicated} are listed more than once')
        return self
