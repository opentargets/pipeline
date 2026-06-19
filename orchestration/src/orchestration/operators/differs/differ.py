import logging
from abc import ABC, abstractmethod

from google.cloud.storage import Client

from orchestration.dags.config.unified_pipeline import UnifiedPipelineConfig


class Differ(ABC):
    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def is_diff(
        self,
        *,
        step_name: str,
        config: UnifiedPipelineConfig,
        client: Client,
    ) -> bool:
        """Determine whether something has changed.

        Returns True if there are differences, False otherwise. This is used to
        decide if a step should run, so as a general rule, if `is_diff`, then the
        step must run.

        Args:
            step_name (str): The name of the step to compare.
            config (UnifiedPipelineConfig): The unified pipeline configuration.
            client (Client): The Google Cloud Storage client used in the differ.

        Returns:
            bool: Whether there are differences in the comparison.
        """
        ...
