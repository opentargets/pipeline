"""Configuration class for the staging DAGs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from orchestration.dags.config.app_config import AppConfig
from orchestration.models.staging_dag import StagingDagConfigModel, StagingEnvironments
from orchestration.utils.utils import read_yaml_config

if TYPE_CHECKING:
    from typing import Any

    from orchestration.models.staging_dag import StagingStep
    from orchestration.operators.dataproc import CustomClusterConfig

# the repository root locally, `/opt` in the airflow containers (see compose.yaml)
PPS_CONFIG_PATH = Path(__file__).resolve().parents[5] / 'pps' / 'config.yaml'

DATAPROC_ENTRYPOINT = 'dataproc_pps_run.py'
"""Main python file of the dataproc jobs running pps steps."""
DATAPROC_INIT_SCRIPT = 'install_pps_on_cluster.sh'
"""Init action installing pps on the dataproc clusters."""


class StagingDagConfig:
    """Configuration class for a staging DAG.

    This class provides the config of a staging DAG and of the pps steps it runs.

    The DAG config is `<dag_dir>/config.yaml`. It is read twice: first to find the
    scratchpad of its selected environment, then rendered with that scratchpad and
    validated against `StagingDagConfigModel`. The same scratchpad overrides the
    scratchpad of `pps/config.yaml`, which the pps steps run with.
    """

    def __init__(self, dag_dir: Path) -> None:
        """Construct the configuration of the staging DAG defined in `dag_dir`."""
        self.logger = logging.getLogger(__name__)
        config_path = dag_dir / 'config.yaml'

        self.dag_id = dag_dir.name
        """The DAG id, which is the name of the DAG directory."""

        # the config can only be rendered once the environment is known, so it is
        # read raw here: its other fields may still hold placeholders
        self.scratchpad = StagingEnvironments.model_validate(read_yaml_config(config_path)).scratchpad
        """The scratchpad of the selected environment."""

        self.dag = AppConfig.from_file(config_path, template_context=self.scratchpad).validate(StagingDagConfigModel)
        """The DAG config, rendered with the scratchpad of the selected environment."""
        self.logger.info(f'loaded staging dag {self.dag_id} in environment {self.dag.env}')

        self.pps = AppConfig.from_file(PPS_CONFIG_PATH)
        """The configuration for pps steps."""

        # Replace config values, as the unified pipeline does for its applications,
        # so pps/config.yaml keeps working for local runs.
        self.pps.config['scratchpad'] = {**self.pps.config.get('scratchpad', {}), **self.scratchpad}
        # relative to the driver's working directory, which dataproc creates per job
        self.pps.config['work_path'] = 'work'
        self.pps.config['log_level'] = 'INFO'

    @property
    def steps(self) -> dict[str, StagingStep]:
        """The steps of the DAG, by name."""
        return self.dag.steps

    @property
    def pps_ref(self) -> str:
        """The git ref of the pps release, url-encoded for pip's `git+<url>@<ref>` syntax."""
        return f'pps%40v{self.dag.pps_version}'

    @property
    def staging_uri(self) -> str:
        """The URI prefix for the files this DAG uploads for a run."""
        return f'{self.dag.staging_bucket.removesuffix("/")}/{self.dag_id}/{self.dag.pps_version}'

    def bin_uri(self, file_name: str) -> str:
        """Return the URI an executable file is uploaded to."""
        return f'{self.staging_uri}/bin/{file_name}'

    def config_uri(self, step_name: str) -> str:
        """Return the URI of the pps configuration file for a step."""
        return f'{self.staging_uri}/config/{step_name}.yaml'

    def step_config(self, step_name: str) -> dict[str, Any]:
        """Return the pps configuration for a step.

        This is `pps/config.yaml`, with the replaced values, restricted to the step.

        Raises:
            ValueError: If the step is not defined in `pps/config.yaml`.
        """
        pps_steps = self.pps.config.get('steps', {})
        if step_name not in pps_steps:
            raise ValueError(f'step {step_name} of dag {self.dag_id} is not defined in {PPS_CONFIG_PATH}')
        return {**self.pps.config, 'steps': {step_name: pps_steps[step_name]}}

    def cluster_config(self, cluster_name: str) -> CustomClusterConfig:
        """Return the configuration of a cluster, set up to install pps when it starts."""
        cluster = self.dag.infrastructure.clusters[cluster_name]
        return cluster.model_copy(
            update={
                'metadata': {**(cluster.metadata or {}), 'PPS_REF': self.pps_ref},
                'init_actions_uris': [self.bin_uri(DATAPROC_INIT_SCRIPT), *(cluster.init_actions_uris or [])],
            }
        )

    def steps_on_cluster(self, cluster_name: str) -> list[str]:
        """Return the names of the steps that run on a cluster."""
        return [name for name, step in self.steps.items() if step.cluster == cluster_name]
