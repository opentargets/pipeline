"""Transformer that extracts what each AACT trial tests with Claude, through the Message Batches API.

The trials are those :py:func:`.aact_trial_extraction.trial_report` gives, each rendered with
:py:func:`.providers.aact_context.render_trial`. Each trial's extraction is a record in the store: the text is
the rendered trial, the model ``aact_trial_extraction_v2``, the revision the digest of the Claude model, the
system prompt, the response schema and the renderer, and the output the JSON of the validated
:py:class:`.schemas.TrialExtractionV2`. Requests go to Claude in batches whose ids are recorded under the store,
so an interrupted run collects the batches it submitted instead of paying for them again. The extraction is
published as a write-once snapshot; a trial that fails every attempt is left out of it.
"""

from __future__ import annotations

import os
from importlib import resources
from typing import TYPE_CHECKING, Any

import polars as pl
from loguru import logger
from otter.storage.synchronous.handle import StorageHandle
from otter.util.errors import TaskRunError
from pydantic import BaseModel, ConfigDict, ValidationError

from pts.transformers.clinical_mira import aact_trial_extraction
from pts.transformers.clinical_mira.moa.claude import MAX_BATCH_REQUESTS, run_batch_cached
from pts.transformers.clinical_mira.model_frames import models_to_frame, polars_schema
from pts.transformers.clinical_mira.providers.aact_context import (
    RENDER_VERSION,
    read_context_tables,
    render_trial,
    trial_context,
)
from pts.transformers.clinical_mira.record_store import (
    ComputeAborted,
    map_texts,
    new_run_id,
    publish_snapshot,
    text_sha256,
)
from pts.transformers.clinical_mira.schemas import TrialExtractionV2
from pts.transformers.clinical_mira.staged_outputs import (
    EXTRACTION_V2,
    EXTRACTION_V2_DEFAULT_MODEL,
    EXTRACTION_V2_MODEL,
    EXTRACTION_V2_PROMPT,
    extraction_v2_revision,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from otter.config.model import Config

WORKSPACE_ID_ENV = 'ANTHROPIC_GOOGLE_CLOUD_WORKSPACE_ID'
"""Environment variable holding the Claude workspace id of the default client."""


class AactTrialExtractionV2Settings(BaseModel):
    """Settings for the aact_trial_extraction_v2 transformer."""

    model_config = ConfigDict(extra='forbid')

    store_uri: str
    """Root of the record stores, an absolute URI. Sits outside the AACT version directory so a trial whose
        rendered text did not change between two AACT versions is not extracted again."""
    snapshot: str
    """The AACT version. Prefixes the id of the runs this step writes and names the batch state file."""
    model: str = EXTRACTION_V2_DEFAULT_MODEL
    """Claude model. Part of the records' revision."""
    effort: str = 'medium'
    """Claude effort."""
    system_prompt: str | None = None
    """Path to a system prompt overriding the packaged one, relative to the release root. Part of the records'
        revision."""
    shard_size: int = 50000
    """Trials per shard, submitted as batches of ``max_batch_requests`` polled together. Each shard is written to
        the store as it finishes."""
    max_attempts: int = 3
    """Attempts per trial within a run. A trial that fails every attempt is left out, and a rerun of the same AACT
        version does not retry it."""
    sample_size: int | None = None
    """Extract a random sample only. For dry runs; leave unset in production."""
    max_batch_requests: int = MAX_BATCH_REQUESTS
    """Requests per submitted batch."""


def _guarded(
    compute: Callable[[list[str]], list[tuple[str | None, str | None]]],
) -> Callable[[list[str]], list[tuple[str | None, str | None]]]:
    """Wrap ``compute`` so it aborts the run when the workspace id variable is unset."""

    def guarded(texts: list[str]) -> list[tuple[str | None, str | None]]:
        if not os.environ.get(WORKSPACE_ID_ENV):
            msg = f'{WORKSPACE_ID_ENV} is not set, so {len(texts)} trials cannot be sent to Claude'
            raise ComputeAborted(msg)
        return compute(texts)

    return guarded


def _system_prompt(settings: AactTrialExtractionV2Settings, config: Config) -> str:
    """Return the system prompt text, from config if overridden and from the package otherwise."""
    if settings.system_prompt:
        text, _ = StorageHandle(settings.system_prompt, config=config).read_text()
        return text
    package, filename = EXTRACTION_V2_PROMPT
    return (resources.files(package) / filename).read_text(encoding='utf-8')


def rendered_trials(archive: str, ids: list[str], scratch_root: str | Path) -> pl.DataFrame:
    """Render the trials ``ids`` of an AACT dump.

    Returns:
        pl.DataFrame: ``id``, ``prompt`` and ``prompt_sha256``, one row per trial of ``ids`` in the dump, by id.
    """
    context = trial_context(**read_context_tables(archive, scratch_root)).filter(pl.col('nct_id').is_in(ids))
    prompts = [(row['nct_id'], render_trial(row)) for row in context.iter_rows(named=True)]
    return pl.DataFrame(prompts, schema={'id': pl.String, 'prompt': pl.String}, orient='row').with_columns(
        prompt_sha256=pl.col('prompt').map_elements(text_sha256, return_dtype=pl.String),
    )


def extraction_v2_compute(
    prompts: pl.DataFrame,
    system_prompt: str,
    settings: AactTrialExtractionV2Settings,
    state_uri: str,
    scratch: Path,
    client: Any = None,
) -> Callable[[list[str]], list[tuple[str | None, str | None]]]:
    """Build the ``compute`` of :py:func:`map_texts` that extracts trials with Claude.

    Args:
        prompts: Output of :py:func:`rendered_trials`.
        system_prompt: The extraction prompt.
        settings: The validated settings.
        state_uri: Where the submitted batch ids are recorded, a path or URI.
        scratch: Local directory for the result cache.
        client: The Anthropic client, ``AnthropicGoogleCloud()`` when ``None``.

    Returns:
        A function from rendered trials to ``(output, error)`` in the order of the trials.
    """
    ids = dict(zip(prompts['prompt'].to_list(), prompts['id'].to_list(), strict=True))
    cache_path = scratch / 'extraction_v2_cache.jsonl'

    def compute(texts: list[str]) -> list[tuple[str | None, str | None]]:
        results, errors = run_batch_cached(
            [(ids[text], text) for text in texts],
            system_prompt=system_prompt,
            response_model=TrialExtractionV2,
            model=settings.model,
            effort=settings.effort,
            cache_path=cache_path,
            state_uri=state_uri,
            client=client,
            max_batch_requests=settings.max_batch_requests,
        )
        return [
            (results[ids[text]].model_dump_json(), None)
            if ids[text] in results
            else (None, errors.get(ids[text], 'missing'))
            for text in texts
        ]

    return compute


def extractions_from_records(records: pl.DataFrame, prompts: pl.DataFrame) -> pl.DataFrame:
    """Build the published extraction from the store's output per rendered trial.

    Args:
        records: ``text`` and ``output`` per rendered trial, as :py:func:`map_texts` returns them.
        prompts: Output of :py:func:`rendered_trials`.

    Returns:
        pl.DataFrame: One row per extracted trial, by id: ``id``, ``prompt_sha256``, then the other fields of
        :py:class:`TrialExtractionV2`. ``id`` is the trial's, not the model's.

    Raises:
        ValueError: If a stored output does not validate, naming its trial.
    """
    joined = records.join(
        prompts.select('id', 'prompt', 'prompt_sha256'), left_on='text', right_on='prompt', maintain_order='left'
    )
    models = []
    for trial_id, output in joined.select('id', 'output').iter_rows():
        try:
            models.append(TrialExtractionV2.model_validate_json(output))
        except ValidationError as exc:
            msg = f'stored extraction of {trial_id} does not validate against TrialExtractionV2: {exc}'
            raise ValueError(msg) from exc
    fields = models_to_frame(models, TrialExtractionV2).drop('id')
    return pl.concat([joined.select('id', 'prompt_sha256'), fields], how='horizontal').sort('id')


def stage_extraction_v2(
    archive: str,
    destination: str,
    settings: AactTrialExtractionV2Settings,
    config: Config,
    client: Any = None,
) -> dict[str, Any]:
    """Extract what the trials of an AACT dump test and publish it as a snapshot.

    Args:
        archive: The AACT dump archive. The tables required are restored inline.
        destination: The directory holding one snapshot per tag.
        settings: The validated settings.
        config: The otter config.
        client: The Anthropic client, ``AnthropicGoogleCloud()`` when ``None``.

    Returns:
        dict[str, Any]: The number of trials, prompts, extractions, computed and failed records, and whether a
        snapshot was published.

    Raises:
        ComputeAborted: If there are trials to extract, no client is given and the workspace id variable is unset.
        TaskRunError: If no trial of a non-empty set was extracted.
    """
    system_prompt = _system_prompt(settings, config)
    revision = extraction_v2_revision(settings.model, system_prompt, RENDER_VERSION)
    tag = EXTRACTION_V2.tag(revision, settings.sample_size)
    polars_schema(TrialExtractionV2)
    store_root = settings.store_uri.rstrip('/')

    report = aact_trial_extraction.trial_report(archive, config.work_path, settings.sample_size)
    ids = report['id'].unique().to_list()
    prompts = rendered_trials(archive, ids, config.work_path)
    logger.info(f'rendered {prompts.height} of {len(ids)} trials')

    compute = extraction_v2_compute(
        prompts,
        system_prompt,
        settings,
        f'{store_root}/batches/{EXTRACTION_V2.name}-{settings.snapshot}-{tag}.json',
        config.work_path,
        client=client,
    )
    mapped = map_texts(
        prompts['prompt'],
        compute if client is not None else _guarded(compute),
        f'{store_root}/{EXTRACTION_V2.name}',
        EXTRACTION_V2_MODEL,
        revision,
        new_run_id(settings.snapshot),
        config,
        max_attempts=settings.max_attempts,
        output_dtype=pl.String(),
        allow_failures=True,
        shard_size=settings.shard_size,
    )
    extractions = extractions_from_records(mapped.frame, prompts)
    if prompts.height and extractions.is_empty():
        msg = f'no trial of {prompts.height} was extracted, so no snapshot is published'
        raise TaskRunError(msg)

    published = publish_snapshot(
        extractions,
        f'{destination.rstrip("/")}/{tag}',
        EXTRACTION_V2.metadata(revision, settings.sample_size),
        config,
    )
    if extractions.height < prompts.height:
        missing = prompts.height - extractions.height
        logger.warning(f'{missing} of {prompts.height} trials ({missing / prompts.height:.1%}) have no extraction')
    return {
        'trials': len(ids),
        'prompts': prompts.height,
        'extracted': extractions.height,
        'computed': mapped.computed,
        'failed': mapped.failed,
        'published': published,
    }


def aact_trial_extraction_v2(
    source: dict[str, str], destination: dict[str, str], settings: dict[str, Any], config: Config
) -> None:
    """Extract what AACT trials test with Claude, reusing prior extractions.

    Args:
        source: Key ``aact``, the AACT dump archive.
        destination: Key ``extraction_v2``, the directory holding one snapshot per tag.
        settings: See :py:class:`AactTrialExtractionV2Settings`.
        config: The otter config.
    """
    s = AactTrialExtractionV2Settings.model_validate(settings)
    stats = stage_extraction_v2(source['aact'], destination['extraction_v2'], s, config)
    logger.info(f'extraction complete: {stats}')
