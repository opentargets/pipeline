"""Configuration class for the staging DAGs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from orchestration.dags.config.app_config import AppConfig
from orchestration.models.staging_dag import StagingDagConfigModel, StagingEnvironments
from orchestration.operators.dataproc import CustomClusterConfig
from orchestration.utils.utils import read_yaml_config

if TYPE_CHECKING:
    from typing import Any

    from orchestration.models.staging_dag import JarFeature, StagingStep

# the repository root locally, `/opt` in the airflow containers (see compose.yaml)
PPS_CONFIG_PATH = Path(__file__).resolve().parents[5] / 'pps' / 'config.yaml'

DATAPROC_ENTRYPOINT = 'dataproc_pps_run.py'
"""Main python file of the dataproc jobs running pps steps."""
DATAPROC_INIT_SCRIPT = 'install_pps_on_cluster.sh'
"""Init action installing pps on the dataproc clusters."""
PPS_IMAGE = 'europe-west1-docker.pkg.dev/open-targets-eu-dev/pipeline/pps'
"""Image the VM steps run, tagged with the pps version (see `.github/workflows/publish.yaml`)."""
VM_CONFIG_PATH = '/config.yaml'
"""Where a VM step's pps config is mounted in its container."""
VM_WORK_PATH = '/mnt/disks/work'
"""Where the work disk of a VM is mounted in its container."""
MAVEN_CENTRAL = 'https://repo1.maven.org/maven2'
"""Where the jars the clusters provide are staged from."""
HAIL_JAR = '/opt/conda/default/lib/python3.11/site-packages/hail/backend/hail-all-spark.jar'
"""The jar of the hail package pps installs on the clusters (see `DATAPROC_INIT_SCRIPT`)."""
SESSION_JAR_FEATURES: dict[str, JarFeature] = {
    'add_delta_lake': 'delta',
    'use_enhanced_bgzip_codec': 'bgzip',
    'start_hail': 'hail',
}
"""The arguments of the pps Session enabling a feature that needs jars, and the feature."""


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

        for step_name, step in self.steps.items():
            if step.cluster is None:
                continue
            provided = self.dag.infrastructure.clusters[step.cluster].jars
            if missing := sorted(self.jar_features(step_name) - set(provided)):
                raise ValueError(
                    f'step {step_name} of dag {self.dag_id} uses {", ".join(missing)}, but its cluster '
                    f'{step.cluster} does not provide the jars: list them in the jars of the cluster'
                )

    @property
    def steps(self) -> dict[str, StagingStep]:
        """The steps of the DAG, by name."""
        return self.dag.steps

    @property
    def pps_ref(self) -> str:
        """The git ref of the pps release, url-encoded for pip's `git+<url>@<ref>` syntax."""
        return f'pps%40v{self.dag.pps_version}'

    @property
    def pps_image(self) -> str:
        """The pps image the VM steps run."""
        return f'{PPS_IMAGE}:{self.dag.pps_version}'

    @property
    def staging_uri(self) -> str:
        """The URI prefix for the files this DAG uploads for a run."""
        # set whenever the dag has pps steps, the only ones uploading files
        assert self.dag.staging_bucket is not None
        return f'{self.dag.staging_bucket.removesuffix("/")}/{self.dag_id}/{self.dag.pps_version}'

    def bin_uri(self, file_name: str) -> str:
        """Return the URI an executable file is uploaded to."""
        return f'{self.staging_uri}/bin/{file_name}'

    def config_uri(self, step_name: str) -> str:
        """Return the URI of the pps configuration file for a step."""
        return f'{self.staging_uri}/config/{step_name}.yaml'

    def step_config(self, step_name: str) -> dict[str, Any]:
        """Return the pps configuration for a step.

        This is `pps/config.yaml`, with the replaced values, restricted to the step. A VM
        step with a work disk works on that disk.

        Raises:
            ValueError: If the step is not defined in `pps/config.yaml`.
        """
        pps_steps = self.pps.config.get('steps', {})
        if step_name not in pps_steps:
            raise ValueError(f'step {step_name} of dag {self.dag_id} is not defined in {PPS_CONFIG_PATH}')
        config = {**self.pps.config, 'steps': {step_name: pps_steps[step_name]}}
        vm = self.steps[step_name].vm if step_name in self.steps else None
        if vm is not None and vm.work_disk_size_gb:
            config['work_path'] = VM_WORK_PATH
        return config

    def jar_features(self, step_name: str) -> set[JarFeature]:
        """Return the features needing jars the pyspark tasks of a pps step use."""
        tasks = self.pps.config.get('steps', {}).get(step_name, [])
        return {
            feature
            for task in tasks
            for argument, feature in SESSION_JAR_FEATURES.items()
            if task.get('session', {}).get(argument)
        }

    def staged_jars(self, cluster_name: str) -> list[tuple[str, str]]:
        """Return the jars a cluster loads from the staging bucket, as `(source url, staged uri)` pairs.

        A jar is staged under its artifact and version, so DAGs pinning different versions
        never overwrite a jar a cluster may be reading.
        """
        # set whenever the dag has pps steps, the only ones running on clusters
        assert self.dag.staging_bucket is not None
        jars_uri = f'{self.dag.staging_bucket.removesuffix("/")}/jars'
        pairs = []
        for coordinates in self.dag.infrastructure.clusters[cluster_name].jars.values():
            for coordinate in coordinates:
                group, artifact, version = coordinate.split(':')
                jar = f'{artifact}-{version}.jar'
                pairs.append((
                    f'{MAVEN_CENTRAL}/{group.replace(".", "/")}/{artifact}/{version}/{jar}',
                    f'{jars_uri}/{jar}',
                ))
        return pairs

    def cluster_config(self, cluster_name: str) -> CustomClusterConfig:
        """Return the configuration of a cluster, set up to install pps when it starts.

        The jars the cluster provides are added to its spark properties, so every job on it
        loads them when it starts.
        """
        cluster = self.dag.infrastructure.clusters[cluster_name]
        properties = dict(cluster.properties or {})
        jars = [*properties.get('spark:spark.jars', '').split(','), *(uri for _, uri in self.staged_jars(cluster_name))]
        if 'hail' in cluster.jars:
            jars.append(HAIL_JAR)
            properties['spark:spark.driver.extraClassPath'] = HAIL_JAR
            properties['spark:spark.executor.extraClassPath'] = './hail-all-spark.jar'
        if jars := [j for j in jars if j]:
            properties['spark:spark.jars'] = ','.join(jars)

        return CustomClusterConfig(
            **cluster.model_dump(exclude={'jars'}),
        ).model_copy(
            update={
                'metadata': {**(cluster.metadata or {}), 'PPS_REF': self.pps_ref},
                'init_actions_uris': [self.bin_uri(DATAPROC_INIT_SCRIPT), *(cluster.init_actions_uris or [])],
                'properties': properties or None,
            }
        )

    def steps_on_cluster(self, cluster_name: str) -> list[str]:
        """Return the names of the steps that run on a cluster."""
        return [name for name, step in self.steps.items() if step.cluster == cluster_name]
