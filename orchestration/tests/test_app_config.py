"""Tests for AppConfig rendering and validation."""

from __future__ import annotations

import pytest
import yaml
from pydantic import BaseModel

from orchestration.dags.config.app_config import AppConfig


def rendered(raw: str, context: dict[str, str] | None = None) -> AppConfig:
    config = AppConfig(raw, yaml.safe_load, template_context=context)
    config._render()
    config._parse()
    return config


def test_render_replaces_placeholders() -> None:
    config = rendered('path: "{{bucket}}/data"', {'bucket': 'gs://b'})

    assert config.config == {'path': 'gs://b/data'}


def test_render_raises_on_placeholders_left_without_value() -> None:
    with pytest.raises(ValueError, match=r'placeholders: \{\{missing\}\}, \{\{other\}\}'):
        rendered('a: "{{bucket}}/{{missing}}"\nb: "{{other}}"', {'bucket': 'gs://b'})


def test_render_keeps_otter_placeholders() -> None:
    config = rendered('path: ${bucket}/data')

    assert config.config == {'path': '${bucket}/data'}


def test_validate_returns_the_model() -> None:
    class Model(BaseModel):
        workers: int

    model = rendered('workers: "{{workers}}"', {'workers': '4'}).validate(Model)

    assert model == Model(workers=4)
