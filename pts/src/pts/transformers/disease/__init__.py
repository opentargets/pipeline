"""The disease transformer."""

from typing import Any

from otter.config.model import Config

from pts.schemas.outputs.disease import Disease
from pts.transformers.disease.disease import disease

__all__ = ['disease']


def validate(
    source: Any,
    destination: str,
    settings: dict[str, Any],
    config: Config,
) -> None:
    """Validate the written `disease` against its dataset model, in the task's validation phase."""
    Disease.validate_dataset(destination)
