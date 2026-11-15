"""Provider-neutral helpers for structured LLM extraction: schema patching and cache keys."""

from __future__ import annotations

import copy
import hashlib
import json
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from pathlib import Path

    from pydantic import BaseModel

KEY_SEPARATOR = '\x1f'


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def patch_schema(schema: dict) -> dict:
    """Patch a JSON schema for structured-output compatibility.

    Structured-output modes require every object node to declare
    ``additionalProperties: false`` and enumerate all properties under
    ``required``. Pydantic's ``model_json_schema`` does not emit these by
    default, so we add them in a single recursive pass.
    """
    schema = copy.deepcopy(schema)

    def _walk(node: object) -> None:
        if isinstance(node, dict):
            if node.get('type') == 'object' or 'properties' in node:
                node['additionalProperties'] = False
                if 'properties' in node:
                    node['required'] = list(node['properties'].keys())
            for v in node.values():
                _walk(v)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(schema)
    return schema


def result_key(input_text: str, system_prompt: str, response_model: type[BaseModel], model: str) -> str:
    """Return the cache key of one call."""
    schema = json.dumps(response_model.model_json_schema(by_alias=True), sort_keys=True)
    parts = (_sha256(input_text), _sha256(system_prompt), _sha256(schema), model)
    return _sha256(KEY_SEPARATOR.join(parts))


def _read_cache(cache_path: Path) -> dict[str, dict]:
    if not cache_path.exists():
        return {}
    cached = {}
    for number, line in enumerate(cache_path.read_text(encoding='utf-8').splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            logger.warning(f'skipping unreadable line {number} of {cache_path}')
            continue
        cached[row['key']] = row['response']
    return cached


def _start_on_a_new_line(cache_path: Path) -> None:
    if cache_path.exists() and cache_path.stat().st_size:
        with cache_path.open('rb') as cache:
            cache.seek(-1, 2)
            torn = cache.read(1) != b'\n'
        if torn:
            with cache_path.open('a', encoding='utf-8') as cache:
                cache.write('\n')
