"""Validate colocalisation, which gentropy writes, in a pts step of its own.

pts does not write colocalisation, so no pts task validates it as it writes it. This transformer reads it
as a source and writes nothing; its `validate` validates it against its model in the
task's validation phase, so a violation fails the task, and L2G does not read it::

    colocalisation_validation:
      - name: transform colocalisation_validation
        transformer: colocalisation_validation
        source:
          colocalisation: output/colocalisation
        destination: {}
"""

from typing import Any

from otter.config.model import Config

from pts.schemas.outputs.colocalisation import Colocalisation


def colocalisation_validation(
    source: dict[str, str],
    destination: dict[str, str],
    settings: dict[str, Any],
    config: Config,
) -> None:
    """Nothing to write: the validation runs in `validate`.

    Raises:
        ValueError: if a destination is given, as the step writes nothing.
    """
    if destination:
        raise ValueError(
            f'colocalisation_validation writes nothing; its destination must be empty, got {destination!r}'
        )


def validate(
    source: dict[str, str],
    destination: dict[str, str],
    settings: dict[str, Any],
    config: Config,
) -> None:
    """Validate the colocalisation gentropy wrote against its dataset model."""
    Colocalisation.validate_dataset(source['colocalisation'])
