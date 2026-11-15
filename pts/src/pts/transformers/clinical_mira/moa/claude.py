"""Structured extraction with Claude through the Message Batches API, with the shared result cache.

Requests are sent as batches (half price, results within 24 hours), each held to the
response model's JSON schema. Submitted batch ids are recorded in ``state_uri`` before
polling, so an interrupted run resumes the batches it already paid for instead of
submitting them again. Results land in a local, append-only cache, keyed on everything
that can change them: the input text, the system prompt, the response schema and the model.

The client defaults to ``anthropic.AnthropicGoogleCloud()`` (Claude Models on Google
Cloud, Anthropic managed), configured by ``ANTHROPIC_GOOGLE_CLOUD_PROJECT`` and
``ANTHROPIC_GOOGLE_CLOUD_WORKSPACE_ID``. The ``anthropic`` package is imported only then.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import time
from typing import TYPE_CHECKING, Any, cast

from loguru import logger
from otter.storage.synchronous.handle import StorageHandle
from otter.util.errors import NotFoundError

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

KEY_SEPARATOR = '\x1f'

USAGE_FIELDS = ('input_tokens', 'output_tokens', 'cache_read_input_tokens', 'cache_creation_input_tokens')
"""Token counts recorded with every cached result, so a run's real cost can be summed."""

CONTENT_ERRORS = frozenset({'refusal', 'max_tokens', 'invalid_json', 'invalid_request_error'})
"""Errors caused by a request or its answer. Any other error is systemic: an expired or canceled request, an
overloaded or failing API."""


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


def read_cache(cache_path: Path) -> dict[str, dict]:
    """The cached response per cache key, skipping unreadable lines."""
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


def start_on_a_new_line(cache_path: Path) -> None:
    """End a cache torn mid-line with a newline, so the next row is readable."""
    if cache_path.exists() and cache_path.stat().st_size:
        with cache_path.open('rb') as cache:
            cache.seek(-1, 2)
            torn = cache.read(1) != b'\n'
        if torn:
            with cache_path.open('a', encoding='utf-8') as cache:
                cache.write('\n')


def cached_results(
    requests: list[tuple[str, str]],
    *,
    system_prompt: str,
    response_model: type[BaseModel],
    model: str,
    cache_path: Path,
) -> tuple[dict[str, BaseModel], dict[str, str]]:
    """Return the cached result of each ``(request_id, input_text)``, calling no model.

    Returns:
        The validated response per cached request id, and no errors.
    """
    cached = read_cache(cache_path)
    results = {}
    for request_id, text in requests:
        key = result_key(text, system_prompt, response_model, model)
        if key in cached:
            results[request_id] = response_model.model_validate(cached[key])
    return results, {}


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


def _error_type(result: Any) -> str:
    """The API error type of an errored batch result, else the result type (``expired``, ``canceled``)."""
    error = getattr(result, 'error', None)
    nested = getattr(error, 'error', None)
    return getattr(nested, 'type', None) or getattr(error, 'type', None) or result.type


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
        request id that failed (an API error type such as ``overloaded_error``, the
        result type ``expired`` or ``canceled``, ``refusal``, ``max_tokens`` or
        ``invalid_json``). Failures are not written to the result cache; whether a
        caller retries them is up to the caller.
    """
    if client is None:
        from anthropic import AnthropicGoogleCloud

        client = cast('Any', AnthropicGoogleCloud())

    cached = read_cache(cache_path)
    keys = {request_id: result_key(text, system_prompt, response_model, model) for request_id, text in requests}
    results = {
        request_id: response_model.model_validate(cached[key]) for request_id, key in keys.items() if key in cached
    }
    by_custom_id = {custom_id(request_id): request_id for request_id, _ in requests}
    errors: dict[str, str] = {}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    start_on_a_new_line(cache_path)

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
                    errors[request_id] = _error_type(item.result)
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

    schema = patch_schema(response_model.model_json_schema(by_alias=True))
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
