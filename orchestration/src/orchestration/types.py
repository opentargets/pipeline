"""Types introduced in the library."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal, TypedDict


class Environment(Enum):
    TEST = 'Test'
    PROD = 'Prod'


class EnvironmentSpec(TypedDict):
    name: Environment
    vars: dict[str, str]


class ConfigNode(TypedDict):
    id: str
    kind: Literal['Task', 'TaskGroup']
    prerequisites: list[str]
    params: dict[str, Any]
    google_batch: Any
    nodes: list[ConfigNode]
    google_batch_index_specs: Any
