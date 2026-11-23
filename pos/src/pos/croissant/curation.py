"""Manual curation of the OT-Croissant Platform instances."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path

from loguru import logger
from pydantic import BaseModel

from pos.croissant.models import InstanceAnnotation


class BaseCuration[T: BaseModel](ABC):
    """Abstract base class for single-file curation tables.

    Subclasses declare which file to load (``curation_path``) and which Pydantic
    model to validate entries against (``_model``). Entries are keyed by their
    ``id`` field.
    """

    def __init__(self) -> None:
        data = json.loads(self.curation_path.read_text())
        self._curation: dict[str, T] = {item['id']: self._model.model_validate(item) for item in data}

    @property
    @abstractmethod
    def curation_path(self) -> Path:
        """Path to the JSON curation file."""

    @property
    @abstractmethod
    def _model(self) -> type[T]:
        """Pydantic model class used to validate each entry."""

    def get(self, id: str, log_level: str = 'WARNING') -> T | None:
        """Return the curation entry for *id*, or ``None`` if not found."""
        entry = self._curation.get(id)
        if entry is None:
            logger.log(
                log_level,
                f"[{type(self).__name__}]: no curation entry found for '{id}'",
            )
        return entry


class InstanceCuration(BaseCuration[InstanceAnnotation]):
    """Curation for Platform instances (public / ppp)."""

    @property
    def curation_path(self) -> Path:
        """Path to the instance curation file."""
        return Path(__file__).parent / 'assets/instance.json'

    @property
    def _model(self) -> type[InstanceAnnotation]:
        """Pydantic model for instance entries."""
        return InstanceAnnotation
