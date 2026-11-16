"""Structured extraction with Claude through the Message Batches API, with the shared result cache.

Requests are sent as batches (half price, results within 24 hours), each held to the
response model's JSON schema. Submitted batch ids are recorded in ``state_uri`` before
polling, so an interrupted run resumes the batches it already paid for instead of
submitting them again. Results land in the same cache as :py:func:`.llm.run_cached`.

The client defaults to ``anthropic.AnthropicGoogleCloud()`` (Claude Models on Google
Cloud, Anthropic managed), configured by ``ANTHROPIC_GOOGLE_CLOUD_PROJECT`` and
``ANTHROPIC_GOOGLE_CLOUD_WORKSPACE_ID``. The ``anthropic`` package is imported only then.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from typing import TYPE_CHECKING, Any, cast

from loguru import logger
from otter.storage.synchronous.handle import StorageHandle
from otter.util.errors import NotFoundError

from pts.transformers.clinical_mira.llm_workflow import _patch_schema
from pts.transformers.clinical_mira.moa.llm import _read_cache, _start_on_a_new_line, result_key

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from pydantic import BaseModel

MAX_TOKENS = 16000
"""Output budget per request; thinking counts towards it."""

MAX_BATCH_REQUESTS = 5000
MAX_BATCH_BYTES = 200_000_000
"""Kept below the API's 100,000 requests and 256 MB per batch."""

POLL_SECONDS = 60

USAGE_FIELDS = ('input_tokens', 'output_tokens', 'cache_read_input_tokens', 'cache_creation_input_tokens')
"""Token counts recorded with every cached result, so a run's real cost can be summed."""


def custom_id(request_id: str) -> str:
    """Return an API-safe, unique ``custom_id`` (``[A-Za-z0-9_-]``, at most 64 characters)."""
    digest = hashlib.sha256(request_id.encode('utf-8')).hexdigest()[:16]
    return f'{re.sub(r"[^A-Za-z0-9_-]", "_", request_id)[:47]}-{digest}'


def _params(text: str, system_prompt: str, schema: dict, model: str, effort: str) -> dict[str, Any]:
    return {
        'model': model,
        'max_tokens': MAX_TOKENS,
        'system': [{'type': 'text', 'text': system_prompt, 'cache_control': {'type': 'ephemeral'}}],
        'messages': [{'role': 'user', 'content': text}],
        'output_config': {'format': {'type': 'json_schema', 'schema': schema}, 'effort': effort},
    }


def _read_state(state_uri: str) -> dict[str, list[str]]:
    try:
        text, _ = StorageHandle(state_uri).read_text()
    except NotFoundError:
        return {'pending': []}
    return json.loads(text)


def _write_state(state_uri: str, state: dict[str, list[str]]) -> None:
    StorageHandle(state_uri).write_text(json.dumps(state, indent=2) + '\n')


def read_usage(cache_path: Path) -> dict[str, dict[str, int]]:
    """The token usage recorded per cache key."""
    usage = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding='utf-8').splitlines():
            if line.strip():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if 'usage' in row:
                    usage[row['key']] = row['usage']
    return usage


def _chunks(requests: list[dict[str, Any]], max_requests: int, max_bytes: int) -> list[list[dict[str, Any]]]:
    chunks: list[list[dict[str, Any]]] = [[]]
    size = 0
    for request in requests:
        request_bytes = len(json.dumps(request))
        if chunks[-1] and (len(chunks[-1]) >= max_requests or size + request_bytes > max_bytes):
            chunks.append([])
            size = 0
        chunks[-1].append(request)
        size += request_bytes
    return [chunk for chunk in chunks if chunk]


def run_batch_cached(
    requests: list[tuple[str, str]],
    *,
    system_prompt: str,
    response_model: type[BaseModel],
    model: str,
    cache_path: Path,
    state_uri: str,
    effort: str = 'medium',
    client: Any = None,
    max_batch_requests: int = MAX_BATCH_REQUESTS,
    max_batch_bytes: int = MAX_BATCH_BYTES,
    poll_seconds: float = POLL_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[dict[str, BaseModel], dict[str, str]]:
    """Run each ``(request_id, input_text)`` through Claude in batches, reusing cached results.

    Returns:
        The validated response per request id that succeeded, and the error type per
        request id that failed (an API error type, ``refusal``, ``max_tokens`` or
        ``invalid_json``). Failures are not cached, so a rerun retries them.
    """
    if client is None:
        from anthropic import AnthropicGoogleCloud

        client = cast('Any', AnthropicGoogleCloud())

    cached = _read_cache(cache_path)
    keys = {request_id: result_key(text, system_prompt, response_model, model) for request_id, text in requests}
    results = {
        request_id: response_model.model_validate(cached[key]) for request_id, key in keys.items() if key in cached
    }
    by_custom_id = {custom_id(request_id): request_id for request_id, _ in requests}
    errors: dict[str, str] = {}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    _start_on_a_new_line(cache_path)

    def collect(batch_id: str) -> None:
        while (batch := client.messages.batches.retrieve(batch_id)).processing_status != 'ended':
            logger.info(f'{batch_id}: {batch.processing_status}, {batch.request_counts.processing} processing')
            sleep(poll_seconds)
        unrequested = 0
        with cache_path.open('a', encoding='utf-8') as cache:
            for item in client.messages.batches.results(batch_id):
                request_id = by_custom_id.get(item.custom_id)
                if request_id is None:
                    unrequested += 1
                    continue
                if request_id in results:
                    continue
                if item.result.type != 'succeeded':
                    errors[request_id] = getattr(getattr(item.result, 'error', None), 'type', item.result.type)
                    continue
                message = item.result.message
                if message.stop_reason in ('refusal', 'max_tokens'):
                    errors[request_id] = message.stop_reason
                    continue
                text = next((b.text for b in message.content if b.type == 'text'), '')
                try:
                    parsed = response_model.model_validate_json(text)
                except ValueError:
                    errors[request_id] = 'invalid_json'
                    continue
                results[request_id] = parsed
                errors.pop(request_id, None)
                response = parsed.model_dump(mode='json', by_alias=True)
                row = {'key': keys[request_id], 'request_id': request_id, 'response': response}
                usage = getattr(message, 'usage', None)
                if usage is not None:
                    row['usage'] = {field: getattr(usage, field, None) or 0 for field in USAGE_FIELDS}
                cache.write(json.dumps(row) + '\n')
        if unrequested:
            logger.warning(f'{batch_id}: {unrequested} results dropped, their custom_id is not among the requests')

    state = _read_state(state_uri)
    for batch_id in list(state['pending']):
        logger.info(f'resuming {batch_id}')
        collect(batch_id)
        state['pending'].remove(batch_id)
        _write_state(state_uri, state)

    pending = [(request_id, text) for request_id, text in requests if request_id not in results]
    logger.info(f'{len(results)} of {len(requests)} requests cached, {len(pending)} to batch with {model}')
    if not pending:
        return results, errors

    schema = _patch_schema(response_model.model_json_schema(by_alias=True))
    batch_requests = [
        {'custom_id': custom_id(request_id), 'params': _params(text, system_prompt, schema, model, effort)}
        for request_id, text in pending
    ]
    for chunk in _chunks(batch_requests, max_batch_requests, max_batch_bytes):
        batch = client.messages.batches.create(requests=chunk)
        state['pending'].append(batch.id)
        _write_state(state_uri, state)
        logger.info(f'submitted {batch.id} with {len(chunk)} requests')
    for batch_id in list(state['pending']):
        collect(batch_id)
        state['pending'].remove(batch_id)
        _write_state(state_uri, state)
    logger.info(f'{len(results)} of {len(requests)} done, {len(errors)} failed')
    return results, errors
