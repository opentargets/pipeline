"""Generic LLM extraction engine.

Accepts a list of pre-built prompts and any Pydantic model for response validation.
``extract_prompts`` returns the outcome per prompt on a client the caller owns;
``run_extraction`` returns a Polars DataFrame of extractions (for parquet output) or None (inspect mode).

Flex processing (service_tier="flex") is enabled by default for ~50% cost
reduction; requests automatically fall back to "auto" on capacity shortage.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import import_module
from importlib.resources import files
from pathlib import Path

import polars as pl
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
        prompts: ``(id, prompt)`` pairs.
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


def _read_text(path: str | Path) -> str:
    """Read a filesystem path or a ``package://`` resource."""
    path_str = str(path)
    if not path_str.startswith('package://'):
        return Path(path).read_text(encoding='utf-8')

    package_path = path_str.removeprefix('package://')
    package, separator, resource_path = package_path.partition('/')
    if not separator or not package or not resource_path:
        raise ValueError("Package resources must use 'package://<package>/<resource>' syntax.")

    resource = files(package)
    for part in resource_path.split('/'):
        resource = resource.joinpath(part)
    return resource.read_text(encoding='utf-8')


def run_extraction(
    prompts: list[dict],
    model_class: str,
    system_prompt_path: str,
    model: str,
    openai_key: str,
    service_tier: str = 'default',
    concurrency: int = 50,
    max_retries: int = 2,
    errors_dir: str | Path | None = None,
) -> pl.DataFrame | None:
    """Extract structured information from pre-built prompts using an LLM.

    Args:
        prompts: List of dicts with keys "id" and "prompt".
        model_class: Dotted path to a Pydantic model class for response
            validation (e.g. "mypackage.models.TrialExtraction").
        system_prompt_path: Filesystem path or ``package://`` URI for the system
            prompt text file.
        model: OpenAI model identifier. Must support the Responses API and,
            if service_tier="flex", flex processing (see OpenAI pricing page).
        openai_key: OpenAI API key.
        service_tier: OpenAI service tier. "flex" cuts costs by ~50% at the
            expense of higher latency; requests fall back automatically to
            "auto" if flex capacity is unavailable. Not available for all models, pass "auto" or "default"
            to opt out.
        concurrency: Maximum number of parallel API calls. Tune against your
            tier's rate limits to avoid sustained 429s.
        max_retries: Retry attempts for transient SDK-level errors (408, 429
            rate-limit, 5xx). Passed directly to the SDK client.
        errors_dir: Directory for errors.jsonl. Defaults to the working
            directory.

    Returns:
        Polars DataFrame with one row per successful extraction, or None in
        single-prompt inspect mode.
    """
    if not openai_key:
        raise ValueError('openai_key must be a non-empty string.')

    model_cls = _import_class(model_class)
    system_prompt = _read_text(system_prompt_path)

    client = AsyncOpenAI(api_key=openai_key, timeout=client_timeout(service_tier), max_retries=max_retries)

    # ── Inspect mode: single prompt ───────────────────────────────────────────
    if len(prompts) == 1:
        entry = prompts[0]
        logger.info('\n── Prompt sent to model ─────────────────────────────────────')
        logger.info(entry['prompt'])
        logger.info('─────────────────────────────────────────────────────────────\n')

        extractions, errors = asyncio.run(
            _run_async(
                prompts,
                model_cls,
                system_prompt,
                model,
                client,
                concurrency,
                service_tier,
            )
        )
        if errors:
            logger.error(f'Error: {errors[0]["error"]}')
            raise RuntimeError(errors[0]['error'])

        logger.info(
            json.dumps(
                json.loads(extractions[0].model_dump_json(exclude_none=True)),
                indent=2,
            )
        )
        return None

    # ── Batch mode ────────────────────────────────────────────────────────────
    logger.info(f'Running extraction: model={model}  concurrency={concurrency}  service_tier={service_tier}')
    extractions, errors = asyncio.run(
        _run_async(prompts, model_cls, system_prompt, model, client, concurrency, service_tier)
    )
    logger.info(f'Extraction complete: {len(extractions)} succeeded, {len(errors)} failed.')

    if errors:
        error_path = Path(errors_dir) / 'errors.jsonl' if errors_dir else Path('errors.jsonl')
        error_path.write_text('\n'.join(json.dumps(e) for e in errors) + '\n')

        for err in errors[:5]:
            logger.error(
                f'Failed  id={err.get("id")!s:<30}  type={err.get("error_type")}  message={err.get("error_message")}'
            )
        if len(errors) > 5:
            logger.error(f'... plus {len(errors) - 5} more errors — see {error_path}')

    return _extractions_to_df(extractions)


# ── Internal helpers ──────────────────────────────────────────────────────────


def _import_class(dotted_path: str) -> type[BaseModel]:
    """Dynamically import a Pydantic model class from a dotted module path."""
    module_path, class_name = dotted_path.rsplit('.', 1)
    return getattr(import_module(module_path), class_name)


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


async def _run_async(
    prompts: list[dict],
    model_cls: type[BaseModel],
    system_prompt: str,
    model: str,
    client: AsyncOpenAI,
    concurrency: int,
    service_tier: str,
) -> tuple[list[BaseModel], list[dict]]:
    pairs = [(entry['id'], entry['prompt']) for entry in prompts]
    outcomes = await _extract_all(client, model_cls, system_prompt, pairs, model, service_tier, concurrency)

    extractions = [o.model for o in outcomes if isinstance(o, Extracted)]
    errors = [
        {'id': prompt_id, 'error': o.message, 'error_type': o.error_type, 'error_message': o.message}
        for (prompt_id, _), o in zip(pairs, outcomes, strict=True)
        if isinstance(o, Failed)
    ]
    return extractions, errors


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


def _extractions_to_df(extractions: Sequence[BaseModel]) -> pl.DataFrame:
    """Serialise a list of Pydantic model instances into a Polars DataFrame.

    Nested structures (lists of models) are preserved as typed
    ``pl.List(pl.Struct(...))`` columns for native Parquet round-trip.
    """
    if not extractions:
        return pl.DataFrame()
    return pl.from_dicts([ext.model_dump() for ext in extractions])
