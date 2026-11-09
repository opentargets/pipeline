"""Configuration class for the staging DAGs."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from orchestration.dags.config.app_config import AppConfig
from orchestration.models.staging_dag import StagingDagConfigModel, StagingEnvironments
from orchestration.models.staging_dag.placeholders import check_rendered

if TYPE_CHECKING:
    from pathlib import Path

    from orchestration.models.staging_dag import StagingClusterConfig, StagingEnvironment


class StagingDagConfig:
    r"""Configuration of the staging DAG defined in `<dag_dir>/config.yaml`.

    The config is read twice with `AppConfig`: first raw, for the validation of the
    environments with `StagingEnvironments`, then rendered as text with the values of the
    selected environment (`{{input.<name>}}`, `{{output}}` and the scratchpad keys), for the
    validation of the rest of the config with `StagingDagConfigModel`. The rendered config
    contains no placeholder: the run folder, `output`, is a literal path of the config, set for
    each run of the DAG, so every path is known when the DAG is parsed.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> dag_dir = Path(tempfile.mkdtemp()) / 'gwas_catalog_top_hits'
        >>> dag_dir.mkdir()
        >>> _ = (dag_dir / 'config.yaml').write_text('''
        ... description: GWAS Catalog curated associations → study index
        ... env: prod
        ... environments:
        ...   prod:
        ...     input: {gwas_catalog: gs://gwas_catalog_inputs}
        ...     output: gs://gwas_catalog_top_hits/prod/20261026T153052
        ...     scratchpad: {gentropy_version: 4.0.0-dev.1, num_workers: '4'}
        ... setup:
        ...   - from: https://raw.githubusercontent.com/opentargets/gentropy/v{{gentropy_version}}/utils/install_dependencies_on_cluster.sh
        ...     to: '{{output}}/bin/install_dependencies_on_cluster.sh'
        ...   - from: https://raw.githubusercontent.com/opentargets/gentropy/v{{gentropy_version}}/src/gentropy/cli.py
        ...     to: '{{output}}/bin/cli.py'
        ... infrastructure:
        ...   clusters:
        ...     otg-tophit-gwascatalog:
        ...       num_workers: '{{num_workers}}'
        ...       metadata: {GENTROPY_REF: 'v{{gentropy_version}}'}
        ...       init_actions_uris: ['{{output}}/bin/install_dependencies_on_cluster.sh']
        ... steps:
        ...   gwas_catalog_top_hit_ingestion:
        ...     kind: dataproc
        ...     definition:
        ...       cluster: otg-tophit-gwascatalog
        ...       main: '{{output}}/bin/cli.py'
        ...       args: |
        ...         step=gwas_catalog_top_hit_ingestion \\
        ...           "step.catalog_study_files=['{{input.gwas_catalog}}/gentroutils/20260802/gwas_catalog_download_studies.tsv']" \\
        ...           step.catalog_studies_out={{output}}/output/study_index \\
        ...           "+step.session.extended_spark_conf={spark.sql.shuffle.partitions: '100'}" \\
        ...           step.session.spark_uri=yarn
        ... ''')
        >>> config = StagingDagConfig(dag_dir)
        >>> config.dag_id, config.environments.env
        ('gwas_catalog_top_hits', 'prod')

        All placeholders are rendered on reading:

        >>> config.dag.infrastructure.clusters['otg-tophit-gwascatalog'].num_workers
        4
        >>> step = config.dag.steps['gwas_catalog_top_hit_ingestion'].definition
        >>> step.main
        'gs://gwas_catalog_top_hits/prod/20261026T153052/bin/cli.py'

        The arguments of `main` are passed as written:

        >>> for arg in step.args:
        ...     print(arg)
        step=gwas_catalog_top_hit_ingestion
        step.catalog_study_files=['gs://gwas_catalog_inputs/gentroutils/20260802/gwas_catalog_download_studies.tsv']
        step.catalog_studies_out=gs://gwas_catalog_top_hits/prod/20261026T153052/output/study_index
        +step.session.extended_spark_conf={spark.sql.shuffle.partitions: '100'}
        step.session.spark_uri=yarn

        The record of the config is in `config/`, and the setup files are copied into
        `bin/`, the location of the cluster and job files:

        >>> config.config_uri
        'gs://gwas_catalog_top_hits/prod/20261026T153052/config/gwas_catalog_top_hits.yaml'
        >>> for copy in config.dag.setup:
        ...     print(copy.to, '<-', copy.source)
        gs://gwas_catalog_top_hits/prod/20261026T153052/bin/install_dependencies_on_cluster.sh <- https://raw.githubusercontent.com/opentargets/gentropy/v4.0.0-dev.1/utils/install_dependencies_on_cluster.sh
        gs://gwas_catalog_top_hits/prod/20261026T153052/bin/cli.py <- https://raw.githubusercontent.com/opentargets/gentropy/v4.0.0-dev.1/src/gentropy/cli.py
        >>> config.cluster_config('otg-tophit-gwascatalog').init_actions_uris
        ['gs://gwas_catalog_top_hits/prod/20261026T153052/bin/install_dependencies_on_cluster.sh']
    """  # noqa: E501

    def __init__(self, dag_dir: Path) -> None:
        """Read and validate the config of the staging DAG defined in `dag_dir`."""
        self.logger = logging.getLogger(__name__)
        config_path = dag_dir / 'config.yaml'

        self.dag_id = dag_dir.name
        """The DAG id, which is the name of the DAG directory."""

        # the config is valid only once rendered with the selected environment, so the
        # environments are read from the raw config first
        self.environments = StagingEnvironments.model_validate(AppConfig.from_file(config_path).config)
        """The environments and the selected one, validated on the raw config."""

        self.app_config = AppConfig.from_file(config_path, template_context=self.environment.template_context)
        """The config, rendered with the values of the selected environment."""
        # the only placeholder check of the config: the rendered text, comments included, is
        # the source of every value and the record of the run
        check_rendered(self.app_config.rendered_config)

        self.dag = StagingDagConfigModel.model_validate({
            k: v for k, v in self.app_config.config.items() if k not in StagingEnvironments.model_fields
        })
        """The validated DAG config, without the environments (validated once, above)."""
        self.logger.info(f'loaded staging dag {self.dag_id} in environment {self.environments.env}')

    @property
    def environment(self) -> StagingEnvironment:
        """The selected environment."""
        return self.environments.environment

    @property
    def config_uri(self) -> str:
        """The URI of the record of the rendered config of the run."""
        return f'{self.environment.config_root}/{self.dag_id}.yaml'

    def rendered_config(self) -> str:
        """Return the rendered config, recorded in the `config/` folder of the run.

        The record is the text of `config.yaml`, rendered, with its comments and layout.
        """
        return self.app_config.rendered_config

    def cluster_config(self, cluster_name: str) -> StagingClusterConfig:
        """Return the config of a cluster."""
        return self.dag.infrastructure.clusters[cluster_name]
