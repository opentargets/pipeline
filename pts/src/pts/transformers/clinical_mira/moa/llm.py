"""Structured LLM calls with a local, append-only result cache.

A result is keyed on everything that can change it: the input text, the system prompt,
the response schema and the model. Only successes are cached, so a rerun retries the
failures and nothing else.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from typing import TYPE_CHECKING

from loguru import logger
from openai import AsyncOpenAI

from pts.transformers.clinical_mira.llm_workflow import Extracted, client_timeout, extract_prompts

if TYPE_CHECKING:
    from pathlib import Path

    from pydantic import BaseModel

KEY_SEPARATOR = '\x1f'


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def result_key(input_text: str, system_prompt: str, response_model: type[BaseModel], model: str) -> str:
    """Return the cache key of one call."""
    schema = json.dumps(response_model.model_json_schema(by_alias=True), sort_keys=True)
    parts = (_sha256(input_text), _sha256(system_prompt), _sha256(schema), model)
    return _sha256(KEY_SEPARATOR.join(parts))


def _read_cache(cache_path: Path) -> dict[str, dict]:
    if not cache_path.exists():
        return {}
    cached = {}
    for line in cache_path.read_text(encoding='utf-8').splitlines():
        if line.strip():
            row = json.loads(line)
            cached[row['key']] = row['response']
    return cached


def run_cached(
    requests: list[tuple[str, str]],
    *,
    system_prompt: str,
    response_model: type[BaseModel],
    model: str,
    cache_path: Path,
    service_tier: str = 'flex',
    concurrency: int = 20,
    chunk_size: int = 100,
    client: AsyncOpenAI | None = None,
) -> tuple[dict[str, BaseModel], dict[str, str]]:
    """Run each ``(request_id, input_text)`` through the model, reusing cached results.

    Returns:
        The validated response per request id that succeeded, and the error type per
        request id that failed.
    """
    return asyncio.run(
        _run(
            requests,
            system_prompt=system_prompt,
            response_model=response_model,
            model=model,
            cache_path=cache_path,
            service_tier=service_tier,
            concurrency=concurrency,
            chunk_size=chunk_size,
            client=client,
        )
    )


async def _run(
    requests: list[tuple[str, str]],
    *,
    system_prompt: str,
    response_model: type[BaseModel],
    model: str,
    cache_path: Path,
    service_tier: str,
    concurrency: int,
    chunk_size: int,
    client: AsyncOpenAI | None,
) -> tuple[dict[str, BaseModel], dict[str, str]]:
    cached = _read_cache(cache_path)
    keys = {request_id: result_key(text, system_prompt, response_model, model) for request_id, text in requests}
    results = {
        request_id: response_model.model_validate(cached[key]) for request_id, key in keys.items() if key in cached
    }
    pending = [(request_id, text) for request_id, text in requests if request_id not in results]
    logger.info(f'{len(results)} of {len(requests)} requests cached, {len(pending)} to run with {model}')

    errors: dict[str, str] = {}
    if not pending:
        return results, errors

    owned = client is None
    active = client or AsyncOpenAI(timeout=client_timeout(service_tier))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        for start in range(0, len(pending), chunk_size):
            chunk = pending[start : start + chunk_size]
            outcomes = await extract_prompts(
                active,
                response_model,
                system_prompt,
                chunk,
                model=model,
                service_tier=service_tier,
                concurrency=concurrency,
            )
            with cache_path.open('a', encoding='utf-8') as cache:
                for request_id, outcome in outcomes.items():
                    if isinstance(outcome, Extracted):
                        results[request_id] = outcome.model
                        response = outcome.model.model_dump(mode='json', by_alias=True)
                        row = {'key': keys[request_id], 'request_id': request_id, 'response': response}
                        cache.write(json.dumps(row) + '\n')
                    else:
                        errors[request_id] = outcome.error_type
            logger.info(f'{min(start + chunk_size, len(pending))} of {len(pending)} run, {len(errors)} failed')
    finally:
        if owned:
            await active.close()
    return results, errors
