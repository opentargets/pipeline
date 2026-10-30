"""Generic LLM extraction engine.

Accepts a list of pre-built prompts and any Pydantic model for response validation.
``extract_prompts`` returns the outcome per prompt on a client the caller owns.

Flex processing (``service_tier="flex"``) costs about half; a request falls
back to ``auto`` on capacity shortage.
"""

from __future__ import annotations

import asyncio
import copy
import logging
from collections.abc import Sequence
from dataclasses import dataclass

from loguru import logger
from openai import APIError, APIStatusError, AsyncOpenAI, BadRequestError, UnprocessableEntityError
from pydantic import BaseModel

logging.getLogger('openai').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)


def _error_type_names(cls: type) -> frozenset[str]:
    return frozenset({cls.__name__}).union(*(_error_type_names(sub) for sub in cls.__subclasses__()))


SYSTEMIC_ERROR_TYPES = (
    _error_type_names(APIError) - _error_type_names(BadRequestError) - _error_type_names(UnprocessableEntityError)
) | {'ServiceTierExhausted'}
"""Workflow error types that come from the API or the transport rather than from a trial's content."""


@dataclass(frozen=True)
class Extracted:
    """A prompt's validated response, its ``id`` set to the prompt's."""

    model: BaseModel


@dataclass(frozen=True)
class Failed:
    """Why a prompt gave no extraction."""

    error_type: str
    message: str


def client_timeout(service_tier: str) -> int:
    """Request timeout in seconds: 15 minutes for flex, 10 otherwise, as OpenAI recommends."""
    return 900 if service_tier == 'flex' else 600


async def extract_prompts(
    client: AsyncOpenAI,
    model_cls: type[BaseModel],
    system_prompt: str,
    prompts: Sequence[tuple[str, str]],
    *,
    model: str,
    service_tier: str,
    concurrency: int,
) -> dict[str, Extracted | Failed]:
    """Extract each prompt with the Responses API, validating against ``model_cls``.

    The caller owns ``client`` and closes it.

    Args:
        client: OpenAI client.
        model_cls: Pydantic model the responses are validated against.
        system_prompt: Instructions sent with every prompt.
        prompts: ``(id, prompt)`` pairs. Ids must be unique.
        model: OpenAI model identifier.
        service_tier: OpenAI service tier; ``flex`` falls back to ``auto`` on capacity shortage.
        concurrency: Maximum number of API calls in flight.

    Returns:
        The outcome per prompt id, in prompt order.
    """
    logger.info(f'Running extraction: model={model}  concurrency={concurrency}  service_tier={service_tier}')
    outcomes = await _extract_all(client, model_cls, system_prompt, prompts, model, service_tier, concurrency)
    failures = [(prompt_id, o) for (prompt_id, _), o in zip(prompts, outcomes, strict=True) if isinstance(o, Failed)]
    logger.info(f'Extraction complete: {len(outcomes) - len(failures)} succeeded, {len(failures)} failed.')
    for prompt_id, failure in failures[:5]:
        logger.error(f'Failed  id={prompt_id!s:<30}  type={failure.error_type}  message={failure.message}')
    if len(failures) > 5:
        logger.error(f'... plus {len(failures) - 5} more errors')
    return {prompt_id: outcome for (prompt_id, _), outcome in zip(prompts, outcomes, strict=True)}


# ── Internal helpers ──────────────────────────────────────────────────────────


def _patch_schema(schema: dict) -> dict:
    """Patch a JSON schema for OpenAI structured-output compatibility.

    OpenAI's structured-output mode requires every object node to declare
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


async def _extract_all(
    client: AsyncOpenAI,
    model_cls: type[BaseModel],
    system_prompt: str,
    prompts: Sequence[tuple[str, str]],
    model: str,
    service_tier: str,
    concurrency: int,
) -> list[Extracted | Failed]:
    semaphore = asyncio.Semaphore(concurrency)
    schema = _patch_schema(model_cls.model_json_schema(by_alias=True))
    return await asyncio.gather(
        *(
            _extract_record(semaphore, client, prompt_id, prompt, model_cls, system_prompt, model, schema, service_tier)
            for prompt_id, prompt in prompts
        )
    )


async def _extract_record(
    semaphore: asyncio.Semaphore,
    client: AsyncOpenAI,
    prompt_id: str,
    prompt: str,
    model_cls: type[BaseModel],
    system_prompt: str,
    model: str,
    schema: dict,
    service_tier: str,
) -> Extracted | Failed:
    """Call the Responses API for a single record.

    Transient errors (408 timeout, rate-limit 429, 5xx) are retried
    automatically by OpenAI.
    One custom layer is added on top: when ``service_tier="flex"`` and the API
    returns a ``resource_unavailable`` 429 (capacity shortage, not a rate
    limit), the request is retried once under ``service_tier="auto"`` so it can
    still complete, at standard cost, rather than failing permanently.
    """
    # If flex is requested, try flex first then fall back to auto on capacity
    # shortage. For any other tier, there is only one attempt.
    tiers_to_try = ['flex', 'auto'] if service_tier == 'flex' else [service_tier]

    for tier in tiers_to_try:
        try:
            async with semaphore:
                response = await client.responses.create(
                    model=model,
                    instructions=system_prompt,
                    input=prompt,
                    text={
                        'format': {
                            'type': 'json_schema',
                            'name': model_cls.__name__,
                            'schema': schema,
                            'strict': True,
                        }
                    },
                    service_tier=tier,  # ty: ignore[invalid-argument-type]
                    store=False,  # extractions are stateless; no persistence needed
                )

            extraction = model_cls.model_validate_json(response.output_text)
            if hasattr(extraction, 'id'):
                extraction.id = prompt_id  # ty: ignore[invalid-assignment]
            return Extracted(extraction)

        except APIStatusError as e:
            # Distinguish flex capacity shortage from ordinary rate-limit 429s.
            # Only the former warrants a tier fallback; the latter is already
            # retried by the SDK.
            is_flex_unavailable = e.status_code == 429 and 'resource_unavailable' in (e.code or '')
            if is_flex_unavailable and tier != tiers_to_try[-1]:
                logger.warning(f'Flex capacity unavailable for id={prompt_id} — retrying with standard processing.')
                continue  # move to "auto"

            return Failed(type(e).__name__, str(e))

        except Exception as e:
            return Failed(type(e).__name__, str(e) or repr(e))

    # Unreachable in normal flow; guards against future loop changes.
    return Failed('ServiceTierExhausted', 'Request failed on both flex and auto tiers.')
