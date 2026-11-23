"""Pydantic models for OT-Croissant curation assets."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class InstanceAnnotation(BaseModel):
    """Annotation for a Platform instance (public / ppp)."""

    model_config = ConfigDict(extra='forbid')

    id: str
    name: str
    description: str
    license: str
    url: str
