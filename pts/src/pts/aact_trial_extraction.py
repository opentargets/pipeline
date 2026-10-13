"""Mira-backed implementation of the :py:mod:`pts.tasks.llm_extract` task."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from typing import TYPE_CHECKING, Any, cast

import polars as pl
from loguru import logger
from mira.provider.aact import extract_clinical_report
from mira.provider.aact.llm_extractor import (
    build_prompts,
    fetch_publications,
    parse_batch_results,
    sample_report,
)
from mira.workflows import llm as llm_workflow
from mira.workflows.llm import run_extraction
from otter.storage.synchronous.handle import StorageHandle
from pydantic import BaseModel

from pts.postgres import read_dump_tables
from pts.result_cache import METADATA_SCHEMA, cache_key, read_cache, write_cache

if TYPE_CHECKING:
    from otter.config.model import Config

    from pts.tasks.llm_extract import LlmExtractSpec

TRIAL_FIELDS = {
    'trialOfficialTitle': 'Official Title',
    'trialDescription': 'Description',
    'trialDetailedDescription': 'Detailed Description',
}
"""Report columns rendered into each prompt, and the labels they get."""

AACT_SCHEMA_NAME = 'ctgov'
AACT_ARCHIVE_MEMBER = 'postgres.dmp'
AACT_TABLES = {
    'studies': [
        'nct_id',
        'overall_status',
        'phase',
        'study_type',
        'start_date',
        'why_stopped',
        'number_of_arms',
        'official_title',
    ],
    'interventions': ['nct_id', 'intervention_type', 'name'],
    'conditions': ['nct_id', 'downcase_name'],
    'study_references': ['nct_id', 'pmid', 'reference_type'],
    'brief_summaries': ['nct_id', 'description'],
    'detailed_descriptions': ['nct_id', 'description'],
}
AACT_ORDER_BY = {'study_references': ['nct_id', 'pmid', 'reference_type']}
LLM_WORKFLOW_PATCH_LOCK = Lock()
"""Serialises temporary patches to Mira's module globals."""


class _AsyncioWithClientCleanup:
    """Delegate asyncio within Mira while closing clients on the same event loop."""

    def __init__(self, module: Any, clients: list[Any]) -> None:
        self._module = module
        self._clients = clients

    def __getattr__(self, name: str) -> Any:
        return getattr(self._module, name)

    def run(self, coroutine: Any, **kwargs: Any) -> Any:
        async def run_with_cleanup() -> Any:
            try:
                return await coroutine
            finally:
                for client in self._clients:
                    await client.close()

        return self._module.run(run_with_cleanup(), **kwargs)


def trial_report(archive: str, scratch_root: str | Path, sample_size: int | None) -> pl.DataFrame:
    """Join the AACT tables into one row per trial.

    This is only the AACT side of the picture. The release dataset of the
    same name unions several other sources and maps entities against the
    disease and molecule indexes, none of which is needed — or available —
    here.
    """
    logger.info(f'restoring AACT tables from {archive}')
    tables = read_dump_tables(
        archive,
        AACT_TABLES,
        schema_name=AACT_SCHEMA_NAME,
        archive_member=AACT_ARCHIVE_MEMBER,
        order_by=AACT_ORDER_BY,
        scratch_root=scratch_root,
    )

    report = extract_clinical_report(
        studies=tables['studies'].select('nct_id', 'study_type', 'phase', 'official_title'),
        interventions=tables['interventions'],
        conditions=tables['conditions'],
        additional_metadata=[
            tables['study_references'],
            tables['brief_summaries'],
            tables['detailed_descriptions'].rename({'description': 'detailed_description'}),
        ],
        aggregation_specs={'pmid': {'group_by': 'nct_id', 'alias': 'literature'}},
    )
    return sample_report(report.df, sample_size)


def prompts_for(report: pl.DataFrame, schema_digest: str, *, max_publications: int, enabled: bool) -> pl.DataFrame:
    """Render one prompt per trial and derive its cache key."""
    publications = fetch_publications(report, max_publications=max_publications, enabled=enabled)
    prompts = build_prompts(report, trial_fields=TRIAL_FIELDS, publications_map=publications)

    return (
        pl
        .DataFrame(prompts, schema={'id': pl.String, 'prompt': pl.String})
        .with_columns(
            prompt_sha256=pl.col('prompt').map_elements(
                lambda p: hashlib.sha256(p.encode('utf-8')).hexdigest(), return_dtype=pl.String
            ),
        )
        .with_columns(
            cache_key=pl.col('id').map_elements(
                lambda trial_id: cache_key(trial_id, schema_digest), return_dtype=pl.String
            ),
        )
    )


def extract(shard: pl.DataFrame, spec: LlmExtractSpec, system_prompt_path: str, errors: list[str]) -> pl.DataFrame:
    """Run the LLM over one shard and return a cache row per trial it extracted.

    ``run_extraction`` returns only what succeeded and logs the rest, so
    trials the model failed on are simply absent from the result, and
    therefore absent from the cache. The next run tries them again. Nothing
    records that they failed — see :py:func:`pts.result_cache.cached_map`.
    Mira's diagnostics are appended to ``errors``.
    """
    prompts = shard.select('id', 'prompt').to_dicts()
    if len(prompts) == 1:
        # run_extraction treats a single prompt as interactive inspect mode
        # and returns None, so never hand it exactly one
        prompts = prompts * 2

    with TemporaryDirectory(prefix='llm-extraction-errors-') as errors_dir:
        extracted = _run_extraction_in_thread(
            prompts=prompts,
            model_class=spec.model_class,
            system_prompt_path=system_prompt_path,
            model=spec.model,
            openai_key=Path(spec.openai_key_path).read_text(encoding='utf-8').strip(),
            service_tier=spec.service_tier,
            concurrency=spec.concurrency,
            max_retries=spec.max_retries,
            errors_dir=errors_dir,
        )
        error_path = Path(errors_dir) / 'errors.jsonl'
        if error_path.exists():
            errors.extend(error_path.read_text(encoding='utf-8').splitlines())
    if extracted is None or extracted.is_empty():
        extracted = pl.DataFrame(schema={'id': pl.String})

    keys = shard.select('id', 'cache_key', 'prompt_sha256')
    succeeded = extracted.unique(subset='id').join(keys, on='id', how='inner')

    missing = shard.height - succeeded.height
    if missing:
        logger.warning(f'{missing} of {shard.height} trials in this shard returned no extraction')

    return succeeded


def seed_legacy_cache(
    prompts: pl.DataFrame,
    legacy_path: str,
    schema_cache_uri: str,
    config: Config,
    snapshot: str,
) -> int:
    """Import legacy Batch API results using the current trial/schema keys.

    This compatibility path is deliberately opt-in and intended for one
    migration run. Failed and malformed legacy records remain absent from
    the cache, so the normal extraction loop retries them.
    """
    logger.info(f'importing legacy AACT batch results from {legacy_path}')
    with TemporaryDirectory(prefix='llm-legacy-batch-') as local:
        if not _stage_batch_results(legacy_path, local, config):
            raise ValueError(f'no *_output.jsonl files found under {legacy_path}')
        legacy = parse_batch_results(local).df
    if legacy.is_empty():
        logger.warning('legacy batch results contained no valid extractions')
        return 0

    imported = legacy.join(prompts.select('id', 'cache_key', 'prompt_sha256'), on='id', how='inner')
    if imported.is_empty():
        logger.warning('no legacy extractions matched the current AACT prompts')
        return 0

    existing = read_cache(schema_cache_uri, config)
    combined = pl.concat([existing, imported], how='diagonal_relaxed')
    combined = combined.with_columns(
        computed_at=pl.coalesce([
            pl.col('computed_at'),
            pl.lit(datetime.now(UTC)).cast(METADATA_SCHEMA['computed_at']),
        ])
    )
    combined = combined.sort('computed_at', descending=True).unique(subset='cache_key', keep='first')
    write_cache(combined, schema_cache_uri, config, snapshot)
    logger.info(f'seeded {imported.height} legacy extractions into the cache')
    return imported.height


def _stage_batch_results(uri: str, local: str, config: Config) -> int:
    """Copy the ``*_output.jsonl`` files under ``uri`` into ``local``, keeping their relative paths."""
    root = StorageHandle(uri, config=config)
    prefix = root.absolute.rstrip('/')
    staged = 0
    for path in root.glob('**/*_output.jsonl'):
        data, _ = StorageHandle(path, config=config).read()
        target = Path(local) / path.removeprefix(prefix).lstrip('/')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        staged += 1
    return staged


def _run_extraction_in_thread(**kwargs: Any) -> pl.DataFrame | None:
    """Call Mira outside Otter's already-running event loop.

    Otter invokes task methods from an asyncio loop, while the synchronous
    Mira's entry point owns its loop via ``asyncio.run``. Running it in
    a short-lived worker thread gives it a thread-local loop without changing
    either library's public API.
    """

    def run_with_full_schema_inference() -> pl.DataFrame | None:
        # Mira's default inference samples only the first 100 model
        # responses. Later validated nested values can then disagree with that
        # inferred schema, losing the whole shard before cached_map can stage it.
        with LLM_WORKFLOW_PATCH_LOCK:
            original = llm_workflow._extractions_to_df

            def to_df(extractions: Sequence[BaseModel]) -> pl.DataFrame:
                return _models_to_dataframe(extractions)

            workflow = cast(Any, llm_workflow)
            original_client_factory = workflow.AsyncOpenAI
            original_asyncio = workflow.asyncio
            clients: list[Any] = []

            def create_client(*args: Any, **client_kwargs: Any) -> Any:
                client = original_client_factory(*args, **client_kwargs)
                clients.append(client)
                return client

            workflow._extractions_to_df = to_df
            workflow.AsyncOpenAI = create_client
            workflow.asyncio = _AsyncioWithClientCleanup(original_asyncio, clients)
            try:
                return run_extraction(**kwargs)
            finally:
                workflow._extractions_to_df = original
                workflow.AsyncOpenAI = original_client_factory
                workflow.asyncio = original_asyncio

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix='llm-extraction') as executor:
        return executor.submit(run_with_full_schema_inference).result()


def _models_to_dataframe(extractions: Sequence[BaseModel]) -> pl.DataFrame:
    """Convert validated model results without losing a whole shard on one row.

    Full inference handles nullable fields that only become populated late in a
    shard. The row-wise fallback protects the checkpoint boundary if a model
    response still exposes a value Polars cannot combine with its neighbours.
    """
    if not extractions:
        return pl.DataFrame()

    rows = [extraction.model_dump() for extraction in extractions]
    try:
        return pl.from_dicts(rows, infer_schema_length=None)
    except pl.exceptions.PolarsError as error:
        logger.warning(
            f'could not combine {len(rows)} extractions at once: {error}; falling back to row-wise conversion'
        )

    converted: list[pl.DataFrame] = []
    for row in rows:
        try:
            converted.append(pl.from_dicts([row], infer_schema_length=None))
        except pl.exceptions.PolarsError as error:
            logger.error(f'dropping one validated extraction during dataframe conversion: {error}')

    if not converted:
        return pl.DataFrame()
    return pl.concat(converted, how='diagonal_relaxed')
