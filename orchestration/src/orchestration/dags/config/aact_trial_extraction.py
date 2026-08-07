"""Configuration class for the AACT trial extraction pipeline. Mirror of `unified_pipeline.py`.

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


class AactTrialExtractionConfig:
    """Configuration for the AACT trial extraction pipeline.

    The pipeline reuses the PIS and PTS applications rather than introducing a
    third one, so this class does what
    :py:class:`~orchestration.dags.config.unified_pipeline.UnifiedPipelineConfig`
    does — load each application's own config and point its ``release_uri`` at
    the run's destination — for the two steps this pipeline needs.

    The destination is ``gs://aact_data/<aact_version>``, keyed on the AACT
    archive rather than on a platform release. The two are on different clocks:
    this pipeline pins its own archive and may run ahead, while a release pins
    whichever archive it consumes and several releases can share one.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        config_path = Path(__file__).parent

        conf = AppConfig.from_file(file_path=config_path / 'aact_trial_extraction.yaml')
        self._steps = conf.get('steps')

        self.bucket: str = conf.get('bucket')
        """Bucket the snapshots and the cache live in."""
        self.aact_version: str = conf.get('aact_version')
        """The AACT monthly archive this run extracts."""
        self.machine_type: str = conf.get('machine_type')
        """Machine type for both steps. The extraction is IO bound, not CPU bound."""
        self.disk_size: int = conf.get('disk_size')
        """Work disk size in GB. Sized for the AACT archive plus its restore."""

        self.snapshot_uri = f'{self.bucket}/{self.aact_version}'
        """Where this run writes.

        Keyed on the AACT archive, not on a platform release. Otter calls this
        ``release_uri`` because that is what it is in a release, and relative
        paths in a step config resolve against it — but nothing about this
        pipeline belongs to a release.
        """

        self.pis = AppConfig.from_file(file_path=config_path.parents[4] / 'pis' / 'config.yaml')
        """The internal configuration for PIS steps."""

        # pis/config.yaml pins the archive the *release* consumes. This pipeline
        # runs on its own schedule and may be ahead of it, so the step runs
        # against this pipeline's version instead.
        self.pis.config['scratchpad']['aact_version'] = self.aact_version
        self.pis.config['release_uri'] = self.snapshot_uri
        self.pis.config['work_path'] = '/mnt/disks/work'
        self.pis.config['log_level'] = 'INFO'
        self.pis.config['pool_size'] = 16

        self.pts = AppConfig.from_file(
            file_path=config_path.parents[4] / 'pts' / 'config.yaml',
            template_context={'release_name': self.aact_version, 'snapshot': self.aact_version},
        )
        """The internal configuration for PTS steps."""

        self.pts.config['release_uri'] = self.snapshot_uri
        self.pts.config['work_path'] = '/mnt/disks/work'
        self.pts.config['log_level'] = 'INFO'
        self.pts.config['pool_size'] = 32

        registry = 'europe-west1-docker.pkg.dev/open-targets-eu-dev/pipeline'
        self.images = {
            'pis': f'{registry}/pis:{conf.get("pis_version")}',
            'pts': f'{registry}/pts:{conf.get("pts_version")}',
        }
        """The image and tag used to run each stage's steps."""

        self.project_id = GCP_PROJECT_PLATFORM
        """The GCP project the steps run in."""

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
        stage_config: AppConfig = getattr(self, stage)
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
        return f'{self.snapshot_uri}/etc/config/{step_name}.yaml'
