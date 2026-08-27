import logging

from google.api_core.exceptions import NotFound as GCSNotFound
from google.cloud.storage import Client

from orchestration.dags.config.unified_pipeline import UnifiedPipelineConfig
from orchestration.operators.differs.differ import Differ
from orchestration.utils.path import GCSPath, IOManager

SUCCESS = 'success'


class StepResultDiffer(Differ):
    """Check whether the step succeeded the last time it ran.

    Without this, clearing a failed step makes it skip itself: `upload_config` runs
    before `run_`, so the config the ConfigDiffer compares against always matches, and
    a failed step records no artifacts, so the ManifestArtifactDiffer walks an empty
    list and reports that all of them exist.

    See opentargets/issues#4511.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def is_diff(
        self,
        *,
        step_name: str,
        config: UnifiedPipelineConfig,
        client: Client,
    ) -> bool:
        """Report a difference unless the manifest records the step as succeeded.

        Anything other than success has to run: `failure` and `aborted` are obvious,
        and `pending` covers a step that started and never recorded an outcome, which
        is what a hung or externally killed step leaves behind.

        Args:
            step_name (str): The name of the step to compare.
            config (UnifiedPipelineConfig): The unified pipeline configuration.
            client (Client): The Google Cloud Storage client used in the differ.

        Returns:
            bool: Whether the step needs to run.
        """
        manifest_uri = config.manifest_uri()
        m = IOManager().resolve(path=manifest_uri)
        if client and isinstance(m, GCSPath):
            m._client = client

        try:
            manifest = m.load()
        except GCSNotFound:
            self.logger.info('manifest not found')
            return True

        step = manifest.get('steps', {}).get(step_name)
        if not step:
            self.logger.info(f'step {step_name} not found in manifest')
            return True

        result = step.get('result')
        if result == SUCCESS:
            return False

        self.logger.info(f'step {step_name} last finished as {result!r}, it must run')
        return True
