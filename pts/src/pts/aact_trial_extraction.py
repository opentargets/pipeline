"""Implementation of the :py:mod:`pts.tasks.llm_extract` task.

Each trial's extraction is a record in a :py:mod:`pts.record_store` store: the
text is the rendered prompt, the model ``aact_llm_extraction``, the revision the
digest of the response schema, and the output the JSON of the validated model.
"""

from __future__ import annotations

import io
import json
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from typing import TYPE_CHECKING, Any, cast

import polars as pl
from loguru import logger
from openai import APIError
from otter.storage.synchronous.handle import StorageHandle
from otter.util.errors import NotFoundError
from pydantic import BaseModel, ValidationError

from pts.postgres import read_dump_tables
from pts.record_store import ComputeAborted, text_sha256, write_run
from pts.transformers.clinical import (
    build_prompts,
    extract_aact_clinical_report,
    llm_workflow,
    run_extraction,
    sample_report,
)

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
EXTRACTION_MODEL = 'aact_llm_extraction'
"""Model name of the extraction records; their model revision is the response schema digest."""
IMPORT_RUN_ID = 'import-extraction-cache'
IMPORT_COMPUTED_AT = datetime(1970, 1, 1, tzinfo=UTC)
"""Run that holds the records imported from the extraction cache, and the time of those cached without one."""
SYSTEMIC_FAILURE_MIN_PROMPTS = 10
"""Trials in an extraction call at or above which getting none back, for systemic reasons only, aborts the run."""


def _error_type_names(cls: type) -> frozenset[str]:
    return frozenset({cls.__name__}).union(*(_error_type_names(sub) for sub in cls.__subclasses__()))


SYSTEMIC_ERROR_TYPES = _error_type_names(APIError) | {'ServiceTierExhausted'}
"""Workflow error types that come from the API or the transport rather than from a trial's content."""
LLM_WORKFLOW_PATCH_LOCK = Lock()
"""Serialises temporary patches to the LLM workflow's module globals."""


class _AsyncioWithClientCleanup:
    """Delegate asyncio within the LLM workflow while closing clients on the same event loop."""

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

    report = extract_aact_clinical_report(
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


def prompts_for(report: pl.DataFrame, schema_digest: str) -> pl.DataFrame:
    """Render one prompt per trial, with the hash of the prompt.

    The prompt is the record store's text, so ``schema_digest`` does not enter
    the result; the store keys it as the model revision instead.
    """
    if 'drugs' in report.columns:
        report = report.with_columns(
            pl.col('drugs').list.eval(pl.element().sort_by(pl.element().struct.field('drugFromSource')))
        )
    prompts = build_prompts(report, trial_fields=TRIAL_FIELDS)

    return pl.DataFrame(prompts, schema={'id': pl.String, 'prompt': pl.String}).with_columns(
        prompt_sha256=pl.col('prompt').map_elements(text_sha256, return_dtype=pl.String),
    )


def extract(shard: pl.DataFrame, spec: LlmExtractSpec, system_prompt_path: str) -> tuple[pl.DataFrame, dict[str, str]]:
    """Run the LLM over one shard and return a row per trial it extracted, and the errors of the rest.

    ``run_extraction`` returns only what succeeded and logs the rest, so
    trials the model failed on are simply absent from the rows. Their errors,
    ``<error_type>: <error_message>`` by trial id, are those the workflow wrote.

    Raises:
        ComputeAborted: If the OpenAI key cannot be read, or a shard of at least
            ``SYSTEMIC_FAILURE_MIN_PROMPTS`` trials yields no extraction at all and
            each trial's error is in ``SYSTEMIC_ERROR_TYPES`` or was not reported.
    """
    try:
        openai_key = Path(spec.openai_key_path).read_text(encoding='utf-8').strip()
    except OSError as exc:
        msg = f'cannot read the OpenAI key at {spec.openai_key_path}: {exc}'
        raise ComputeAborted(msg) from exc

    prompts = shard.select('id', 'prompt').to_dicts()
    if len(prompts) == 1:
        # run_extraction treats a single prompt as interactive inspect mode
        # and returns None, so never hand it exactly one
        prompts = prompts * 2

    errors: dict[str, str] = {}
    error_types: dict[str, str] = {}
    with TemporaryDirectory(prefix='llm-extraction-errors-') as errors_dir:
        extracted = _run_extraction_in_thread(
            prompts=prompts,
            model_class=spec.model_class,
            system_prompt_path=system_prompt_path,
            model=spec.model,
            openai_key=openai_key,
            service_tier=spec.service_tier,
            concurrency=spec.concurrency,
            max_retries=spec.max_retries,
            errors_dir=errors_dir,
        )
        error_path = Path(errors_dir) / 'errors.jsonl'
        if error_path.exists():
            for line in error_path.read_text(encoding='utf-8').splitlines():
                if line.strip():
                    error = json.loads(line)
                    errors[str(error.get('id'))] = f'{error.get("error_type")}: {error.get("error_message")}'
                    error_types[str(error.get('id'))] = str(error.get('error_type'))
    if extracted is None or extracted.is_empty():
        extracted = pl.DataFrame(schema={'id': pl.String})

    trials = shard.select('id').unique()
    succeeded = extracted.unique(subset='id', keep='first', maintain_order=True).join(trials, on='id', how='semi')
    systemic = all(error_types.get(i) in (None, *SYSTEMIC_ERROR_TYPES) for i in trials['id'])
    if succeeded.is_empty() and trials.height >= SYSTEMIC_FAILURE_MIN_PROMPTS and systemic:
        examples = '; '.join(list(errors.values())[:3]) or 'no errors reported'
        msg = f'none of {trials.height} trials returned an extraction, all for systemic reasons, e.g. {examples}'
        raise ComputeAborted(msg)

    missing = shard.height - succeeded.height
    if missing:
        logger.warning(f'{missing} of {shard.height} trials in this shard returned no extraction')

    return succeeded, errors


def extraction_compute(
    prompts: pl.DataFrame, spec: LlmExtractSpec, system_prompt_path: str
) -> Callable[[list[str]], list[tuple[str | None, str | None]]]:
    """Return the record store compute that extracts the trials of a list of prompts.

    Each prompt maps to the JSON of its trial's extraction, or to a failure
    carrying the workflow's error for the trial, ``no extraction`` if it gave none.
    """
    ids = dict(zip(prompts['prompt'].to_list(), prompts['id'].to_list(), strict=True))

    def compute(texts: list[str]) -> list[tuple[str | None, str | None]]:
        shard = pl.DataFrame(
            {'id': [ids[t] for t in texts], 'prompt': texts}, schema={'id': pl.String, 'prompt': pl.String}
        )
        extracted, errors = extract(shard, spec, system_prompt_path)
        rows = {row['id']: row for row in extracted.to_dicts()}
        return [
            (json.dumps(rows[i]), None) if i in rows else (None, errors.get(i, 'no extraction')) for i in shard['id']
        ]

    return compute


def extractions_from_records(
    records: pl.DataFrame, prompts: pl.DataFrame, model_class: type[BaseModel]
) -> pl.DataFrame:
    """Build the published extraction from the store's output per prompt.

    Args:
        records: ``text`` and ``output`` per prompt, as :py:func:`pts.record_store.map_texts` returns them.
        prompts: Output of :py:func:`prompts_for`.
        model_class: The model each output is validated against. It has an ``id`` field.

    Returns:
        pl.DataFrame: One row per extracted trial: the model's fields, then ``prompt_sha256``.

    Raises:
        ValueError: If a stored output does not validate, naming its trial.
    """
    joined = records.join(
        prompts.select('id', 'prompt', 'prompt_sha256'), left_on='text', right_on='prompt', maintain_order='left'
    )
    models = []
    for trial_id, output in joined.select('id', 'output').iter_rows():
        try:
            models.append(model_class.model_validate_json(output))
        except ValidationError as exc:
            msg = f'stored extraction of {trial_id} does not validate against {model_class.__name__}: {exc}'
            raise ValueError(msg) from exc
    extractions = _models_to_dataframe(models)
    if extractions.is_empty():
        return pl.DataFrame(schema={'id': pl.String, 'prompt_sha256': pl.String})
    hashes = joined.select('id', 'prompt_sha256').unique(subset='id', keep='first', maintain_order=True)
    return extractions.join(hashes, on='id', how='left', maintain_order='left')


def import_extraction_cache(
    cache_uri: str, prompts: pl.DataFrame, store_uri: str, schema_digest: str, config: Config
) -> int:
    """Import an extraction cache of one response schema into the record store, once.

    Every cached trial still in ``prompts`` becomes an ``ok`` record whose text is
    that trial's current prompt. Of several rows for one trial, the one cached
    for the current prompt wins, then the newest.

    Args:
        cache_uri: The cache's namespace for ``schema_digest``: ``latest.json``
            pointing at ``snapshots/<snapshot>/rows.parquet``.
        prompts: Output of :py:func:`prompts_for`.
        store_uri: Root of the store, an absolute URI.
        schema_digest: Digest of the response schema, the records' model revision.
        config: The otter config, used to resolve storage.

    Returns:
        int: The number of records written, 0 if the import run already exists.

    Raises:
        ValueError: If ``cache_uri`` is not the namespace of ``schema_digest``.
    """
    if not cache_uri.rstrip('/').endswith(f'/{schema_digest}'):
        msg = f'{cache_uri} is not the extraction cache namespace of schema {schema_digest}'
        raise ValueError(msg)

    try:
        StorageHandle(f'{store_uri}/runs/{IMPORT_RUN_ID}/manifest.json', config=config).read_text()
    except NotFoundError:
        pass
    else:
        logger.info(f'{IMPORT_RUN_ID} already imported into {store_uri}')
        return 0

    pointer, _ = StorageHandle(f'{cache_uri}/latest.json', config=config).read_text()
    snapshot = json.loads(pointer)['snapshot']
    data, _ = StorageHandle(f'{cache_uri}/snapshots/{snapshot}/rows.parquet', config=config).read()
    rows = pl.read_parquet(io.BytesIO(data))
    if 'prompt_sha256' not in rows.columns:
        rows = rows.with_columns(prompt_sha256=pl.lit(None, dtype=pl.String))

    current = prompts.select('id', 'prompt', current_sha256='prompt_sha256').unique(subset='id')
    adopted = (
        rows
        .join(current, on='id')
        .with_columns(is_current=(pl.col('prompt_sha256') == pl.col('current_sha256')).fill_null(False))
        .sort(['is_current', 'computed_at'], descending=True, nulls_last=True)
        .unique(subset='id', keep='first', maintain_order=True)
        .sort('id')
    )
    outputs = adopted.drop('cache_key', 'prompt_sha256', 'computed_at', 'prompt', 'current_sha256', 'is_current')
    records = adopted.select(
        text_sha256=pl.col('current_sha256'),
        text=pl.col('prompt'),
        model=pl.lit(EXTRACTION_MODEL),
        model_revision=pl.lit(schema_digest),
        output=pl.Series([json.dumps(row) for row in outputs.to_dicts()], dtype=pl.String),
        status=pl.lit('ok'),
        error=pl.lit(None, dtype=pl.String),
        computed_at=pl.col('computed_at').fill_null(IMPORT_COMPUTED_AT),
        run_id=pl.lit(IMPORT_RUN_ID),
    )
    write_run(records, store_uri, IMPORT_RUN_ID, config, output_dtype=pl.String())
    return records.height


def _run_extraction_in_thread(**kwargs: Any) -> pl.DataFrame | None:
    """Call the LLM workflow outside Otter's already-running event loop.

    Otter invokes task methods from an asyncio loop, while the synchronous
    workflow's entry point owns its loop via ``asyncio.run``. Running it in
    a short-lived worker thread gives it a thread-local loop without changing
    Otter's or the workflow's API.
    """

    def run_with_full_schema_inference() -> pl.DataFrame | None:
        # The workflow's default inference samples only the first 100 model
        # responses. Later validated nested values can then disagree with that
        # inferred schema, losing the whole shard before the record store can write it.
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
