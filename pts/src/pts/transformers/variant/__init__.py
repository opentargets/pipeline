"""The variant transformer."""

from typing import Any

from otter.config.model import Config

from pts.schemas.outputs.variant import Variant
from pts.transformers.variant.variant import variant

__all__ = ['variant']


def validate(
    source: Any,
    destination: dict[str, str],
    settings: dict[str, Any],
    config: Config,
) -> None:
    """Validate the written `variant` against its dataset model, in the task's validation phase."""
    Variant.validate_dataset(destination['output'])
