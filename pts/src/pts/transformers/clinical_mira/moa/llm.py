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


def run_cached(
    requests: list[tuple[str, str]],
    *,
    system_prompt: str,
    response_model: type[BaseModel],
    model: str,
    cache_path: Path,
    service_tier: str = 'flex',
    concurrency: int = 20,
    chunk_size: int | None = None,
    client: AsyncOpenAI | None = None,
    cached_only: bool = False,
) -> tuple[dict[str, BaseModel], dict[str, str]]:
    """Run each ``(request_id, input_text)`` through the model, reusing cached results.

    Results are written after every chunk of ``chunk_size`` requests (by default the
    concurrency), so an interruption loses at most one chunk of finished calls. With
    ``cached_only`` nothing is sent and only cached results are returned.

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
            chunk_size=chunk_size or concurrency,
            client=client,
            cached_only=cached_only,
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
    cached_only: bool,
) -> tuple[dict[str, BaseModel], dict[str, str]]:
    cached = _read_cache(cache_path)
    keys = {request_id: result_key(text, system_prompt, response_model, model) for request_id, text in requests}
    results = {
        request_id: response_model.model_validate(cached[key]) for request_id, key in keys.items() if key in cached
    }
    pending = [(request_id, text) for request_id, text in requests if request_id not in results]
    logger.info(f'{len(results)} of {len(requests)} requests cached, {len(pending)} to run with {model}')

    errors: dict[str, str] = {}
    if not pending or cached_only:
        return results, errors

    owned = client is None
    active = client or AsyncOpenAI(timeout=client_timeout(service_tier))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    _start_on_a_new_line(cache_path)
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
