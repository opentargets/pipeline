"""Structured LLM calls with a local, append-only result cache.

A result is keyed on everything that can change it: the input text, the system prompt,
the response schema and the model. Only successes are cached, so a rerun retries the
failures and nothing else.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from typing import TYPE_CHECKING, Any, Literal

from loguru import logger
from openai import AsyncOpenAI

from pts.transformers.clinical_mira.llm_workflow import (
    Extracted,
    Failed,
    _patch_schema,
    client_timeout,
    extract_prompts,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
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


CHAT_MAX_TOKENS = 16000
"""Output budget per chat-completions call, including the model's reasoning."""


async def chat_extract(
    client: Any,
    model_cls: type[BaseModel],
    system_prompt: str,
    prompts: Sequence[tuple[str, str]],
    *,
    model: str,
    concurrency: int,
) -> dict[str, Extracted | Failed]:
    """Extract each prompt through a chat-completions endpoint held to ``model_cls``'s JSON schema."""
    semaphore = asyncio.Semaphore(concurrency)
    response_format = {
        'type': 'json_schema',
        'json_schema': {
            'name': model_cls.__name__,
            'schema': _patch_schema(model_cls.model_json_schema(by_alias=True)),
            'strict': True,
        },
    }

    async def one(prompt: str) -> Extracted | Failed:
        try:
            async with semaphore:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': prompt}],
                    response_format=response_format,
                    max_tokens=CHAT_MAX_TOKENS,
                )
            return Extracted(model_cls.model_validate_json(response.choices[0].message.content or ''))
        except Exception as e:
            return Failed(type(e).__name__, str(e) or repr(e))

    outcomes = await asyncio.gather(*(one(prompt) for _, prompt in prompts))
    return {prompt_id: outcome for (prompt_id, _), outcome in zip(prompts, outcomes, strict=True)}


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
    api: Literal['responses', 'chat'] = 'responses',
    client_factory: Callable[[], Any] | None = None,
) -> tuple[dict[str, BaseModel], dict[str, str]]:
    """Run each ``(request_id, input_text)`` through the model, reusing cached results.

    ``api`` picks the OpenAI Responses API or a chat-completions endpoint (such as Vertex
    AI's). ``client_factory``, when given, builds a fresh client for every chunk, so a
    short-lived access token is renewed during a long run.

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
            api=api,
            client_factory=client_factory,
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
    api: Literal['responses', 'chat'],
    client_factory: Callable[[], Any] | None,
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

    def make_client() -> Any:
        if client is not None:
            return client
        if client_factory is not None:
            return client_factory()
        return AsyncOpenAI(timeout=client_timeout(service_tier))

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    _start_on_a_new_line(cache_path)
    for start in range(0, len(pending), chunk_size):
        chunk = pending[start : start + chunk_size]
        active = make_client()
        try:
            if api == 'chat':
                outcomes = await chat_extract(
                    active, response_model, system_prompt, chunk, model=model, concurrency=concurrency
                )
            else:
                outcomes = await extract_prompts(
                    active,
                    response_model,
                    system_prompt,
                    chunk,
                    model=model,
                    service_tier=service_tier,
                    concurrency=concurrency,
                )
        finally:
            if active is not client:
                await active.close()
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
    return results, errors
