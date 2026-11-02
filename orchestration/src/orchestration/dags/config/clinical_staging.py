"""Configuration class for the clinical staging pipeline. Mirror of `unified_pipeline.py`.

Required to initialise PIS and PTS steps.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from orchestration.dags.config.app_config import AppConfig
from orchestration.utils.common import GCP_PROJECT_PLATFORM

if TYPE_CHECKING:
    from typing import Any


class ClinicalStagingConfig:
    """Configuration for the clinical staging pipeline.

    The pipeline reuses the PIS and PTS applications rather than introducing a
    third one, so this class does what
    :py:class:`~orchestration.dags.config.unified_pipeline.UnifiedPipelineConfig`
    does — load each application's own config and point its ``release_uri`` at
    the run's destination — for the steps this pipeline needs.

    Every step belongs to a source, ``aact`` or ``clinical_mining``, and writes
    under ``gs://clinical_staging/<source>/<version>``.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        config_path = Path(__file__).parent

        conf = AppConfig.from_file(file_path=config_path / 'clinical_staging.yaml')
        self._steps = conf.get('steps')

        self.bucket: str = conf.get('bucket')
        """Bucket the snapshots and the record stores live in."""
        self.aact_version: str = conf.get('aact_version')
        """The AACT monthly archive this run stages."""
        self.clinical_mining_version: str = conf.get('clinical_mining_version')
        """The clinical mining inputs this run stages."""
        self.machine_type: str = conf.get('machine_type')
        """Default machine type. The extraction is IO bound, not CPU bound; a step may override it."""
        self.disk_size: int = conf.get('disk_size')
        """Work disk size in GB. Sized for the AACT archive plus its restore."""
        self.boot_disk_size: int = conf.get('boot_disk_size')
        """Boot disk size in GB. Sized for the PIS and PTS container images."""

        self.versions = {'aact': self.aact_version, 'clinical_mining': self.clinical_mining_version}
        """The version of each source."""

        pis_path = config_path.parents[4] / 'pis' / 'config.yaml'
        pts_path = config_path.parents[4] / 'pts' / 'config.yaml'

        self.pis: dict[str, AppConfig] = {}
        """The internal configuration for PIS steps, per source."""
        self.pts: dict[str, AppConfig] = {}
        """The internal configuration for PTS steps, per source."""
        for source, version in self.versions.items():
            pis = AppConfig.from_file(file_path=pis_path)
            pis.config['scratchpad'][f'{source}_version'] = version
            pis.config['release_uri'] = self.source_uri(source)
            pis.config['work_path'] = '/mnt/disks/work'
            pis.config['log_level'] = 'INFO'
            pis.config['pool_size'] = 16
            self.pis[source] = pis

            pts = AppConfig.from_file(
                file_path=pts_path,
                template_context={'release_name': version, 'snapshot': version},
            )
            pts.config['release_uri'] = self.source_uri(source)
            pts.config['work_path'] = '/mnt/disks/work'
            pts.config['log_level'] = 'INFO'
            pts.config['pool_size'] = 32
            self.pts[source] = pts

        registry = 'europe-west1-docker.pkg.dev/open-targets-eu-dev/pipeline'
        self.images = {
            'pis': f'{registry}/pis:{conf.get("pis_version")}',
            'pts': f'{registry}/pts:{conf.get("pts_version")}',
        }
        """The image and tag used to run each stage's steps."""

        self.project_id = GCP_PROJECT_PLATFORM
        """The GCP project the steps run in."""

    def source_uri(self, source: str) -> str:
        """Return where a source writes: ``<bucket>/<source>/<version>``.

        Args:
            source: ``aact`` or ``clinical_mining``.

        Returns:
            str: The version directory of the source.
        """
        return f'{self.bucket}/{source}/{self.versions[source]}'

    def release_uri(self, step_name: str) -> str:
        """Return the version directory a step reads and writes.

        Args:
            step_name: The name of the step, in the form ``{stage}_{step}``.

        Returns:
            str: The URI under which the step's source is staged.
        """
        return self.source_uri(self.step_definition(step_name)['source'])

    def steps(self) -> list[str]:
        """Return the steps in the pipeline, in the form ``{stage}_{step}``.

        Returns:
            list[str]: The list of step names.
        """
        return list(self._steps.keys())

    def step_definition(self, step_name: str) -> dict[str, Any]:
        """Return the definition of a step: its dependencies and any secrets it needs.

        Args:
            step_name: The name of the step, in the form ``{stage}_{step}``.

        Returns:
            dict[str, Any]: The definition of the step.
        """
        # can't put the default in the get, as the content can actually be None
        # and that will not be replaced by the default
        return self._steps.get(step_name) or {}

    def machine_type_for(self, step_name: str) -> str:
        """Return the machine type a step runs on.

        Args:
            step_name: The name of the step, in the form ``{stage}_{step}``.

        Returns:
            str: The step's own ``machine_type`` if it declares one, else the global one.
        """
        return self.step_definition(step_name).get('machine_type') or self.machine_type

    def step_config(self, step_name: str) -> dict[str, Any]:
        """Return the configuration to upload for a step.

        The step's own definition lives in its application's config file, keyed
        by the part of the name after the stage prefix.

        Args:
            step_name: The name of the step, in the form ``{stage}_{step}``.

        Returns:
            dict[str, Any]: The application config, with only this step under ``steps``.
        """
        stage, step = step_name.split('_', 1)
        stage_config: AppConfig = getattr(self, stage)[self.step_definition(step_name)['source']]
        return {
            **stage_config.config,
            'steps': {step: stage_config.config.get('steps', {}).get(step, {})},
        }

    def step_image(self, step_name: str) -> str:
        """Return the container image that runs a step.

        Args:
            step_name: The name of the step, in the form ``{stage}_{step}``.

        Returns:
            str: The image and tag.
        """
        stage, _ = step_name.split('_', 1)
        return self.images[stage]

    def step_env_vars(self, step_name: str) -> dict[str, str]:
        """Return the environment variables a step's container needs.

        Both applications take the same pair, under their own prefix.

        Args:
            step_name: The name of the step, in the form ``{stage}_{step}``.

        Returns:
            dict[str, str]: The environment variables.
        """
        stage, step = step_name.split('_', 1)
        return {
            f'{stage.upper()}_STEP': step,
            f'{stage.upper()}_CONFIG_PATH': '/config.yaml',
        }

    def config_uri(self, step_name: str) -> str:
        """Return the URI the config for a step is uploaded to.

        Args:
            step_name: The step name.

        Returns:
            str: The URI of the configuration file for the step.
        """
        return f'{self.release_uri(step_name)}/etc/config/{step_name}.yaml'
