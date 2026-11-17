"""The disease transformer."""

from typing import Any

from otter.config.model import Config

from pts.schemas.outputs.disease import Disease
from pts.task_validation import validate_outputs
from pts.transformers.disease.disease import disease

__all__ = ['disease']


def disease_validate(
    source: Any,
    destination: str,
    settings: dict[str, Any],
    config: Config,
) -> None:
    """Validate the written `disease` against its dataset model, in the task's validation phase."""
    validate_outputs({destination: Disease})
