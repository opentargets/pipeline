"""Task that extracts structured drug and disease evidence from clinical trials with an LLM.

The extraction itself lives in ``clinical_mining``; this task is the part that
makes it repeatable — it builds one prompt per trial, asks
:py:mod:`pts.result_cache` for the ones that have not been extracted yet, and
publishes the result as a typed parquet the release can read directly.

Cache invalidation is derived, never declared. The key hashes the trial prompt
and the JSON schema the model is held to. The model and system instructions are
operational choices and intentionally do not invalidate accepted results.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from importlib import import_module, resources
from pathlib import Path
from typing import Any, Self

import polars as pl
from clinical_mining.provider.aact import extract_clinical_report
from clinical_mining.provider.aact.llm_extractor import (
    build_prompts,
    fetch_publications,
    parse_batch_results,
    sample_report,
)
from clinical_mining.workflows.llm import run_extraction
from loguru import logger
from otter.manifest.model import Artifact
from otter.storage.synchronous.handle import StorageHandle
from otter.task.model import Spec, Task, TaskContext
from otter.task.task_reporter import report
from otter.util.errors import TaskValidationError
from pydantic import BaseModel

from pts.postgres import read_dump_tables
from pts.result_cache import METADATA_SCHEMA, cache_key, cached_map, read_cache, write_cache

DEFAULT_SYSTEM_PROMPT = ('clinical_mining.prompts', 'aact_llm.txt')
"""Packaged system prompt, versioned with the pinned ``clinical-mining`` dependency."""

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


class PublicationsSpec(BaseModel):
    """Whether to enrich prompts with Europe PMC abstracts."""

    enabled: bool = False
    """Enrich prompts with abstracts.

        .. warning:: This fetches abstracts for *every* trial, not only the ones
            that missed the cache, because the abstract is part of the prompt and
            so part of the cache key. It is the one part of this step that does
            not get cheaper as the cache fills. Give it its own
            :py:func:`pts.result_cache.cached_map` keyed on pmid before turning it on
            for a full run."""
    max_publications: int = 1
    """Abstracts per trial."""


class LlmExtractSpec(Spec):
    """Configuration fields for the llm_extract task."""

    source: dict[str, str]
    """AACT dump archive. The task restores the required tables inline."""
    destination: dict[str, str]
    """Where to publish. Keys: ``prompts``, ``extraction``."""
    cache_uri: str
    """Root of the extraction cache, an absolute URI. Must sit outside the
        dataset version directory so a trial whose text did not change between
        two AACT snapshots stays a cache hit."""
    snapshot: str
    """Name for the cache snapshot this run writes, and for its staging area.
        Reuse it to resume a failed run; change it to start clean. The DAG sets
        it from the Airflow run id."""
    openai_key_path: str = '/var/run/secrets/openai-api-key'
    """Path the OpenAI key is mounted at, injected from Secret Manager by the
        DAG. Never put the key itself in config."""
    model: str = 'gpt-5-nano-2025-08-07'
    """OpenAI model. Not part of the cache key."""
    model_class: str = 'clinical_mining.schemas.ClinicalReportExtractionSchema'
    """Dotted path to the pydantic model the response is validated against. Its
        JSON schema is part of the cache key."""
    system_prompt: str | None = None
    """Path to a system prompt overriding the packaged one, relative to the
        release root. Its contents are not part of the cache key."""
    service_tier: str = 'flex'
    """OpenAI service tier. ``flex`` is roughly half price for higher latency,
        with an automatic fallback to standard when flex capacity runs out."""
    concurrency: int = 50
    """Parallel API calls in flight."""
    max_retries: int = 2
    """SDK-level retries for transient errors."""
    shard_size: int = 2000
    """Trials per shard. Each shard is staged as it finishes, so a retry of this
        task resumes rather than paying for the same calls twice."""
    sample_size: int | None = None
    """Extract a random sample only. For dry runs; leave unset in production."""
    publications: PublicationsSpec = PublicationsSpec()
    """Europe PMC enrichment."""
    legacy_batch_results: str | None = None
    """Optional one-off directory of legacy OpenAI Batch API results to seed the cache."""
    legacy_import_only: bool = False
    """Seed legacy results and stop before making any new API calls."""


class LlmExtract(Task):
    """Extract structured evidence from AACT trials, reusing prior extractions."""

    def __init__(
        self,
        spec: LlmExtractSpec,
        context: TaskContext,
    ) -> None:
        super().__init__(spec, context)
        self.spec: LlmExtractSpec
        self.stats: dict[str, Any] = {}

    def _system_prompt(self) -> str:
        """Return the system prompt text, from config if overridden and from the package otherwise."""
        if self.spec.system_prompt:
            text, _ = StorageHandle(self.spec.system_prompt, config=self.context.config).read_text()
            return text
        package, filename = DEFAULT_SYSTEM_PROMPT
        return (resources.files(package) / filename).read_text(encoding='utf-8')

    def _trial_report(self) -> pl.DataFrame:
        """Join the AACT tables into one row per trial.

        This is only the AACT side of the picture. The release dataset of the
        same name unions several other sources and maps entities against the
        disease and molecule indexes, none of which is needed — or available —
        here.
        """
        archive = StorageHandle(str(self.spec.source['aact']), config=self.context.config).absolute
        logger.info(f'restoring AACT tables from {archive}')
        tables = read_dump_tables(
            archive,
            AACT_TABLES,
            schema_name=AACT_SCHEMA_NAME,
            archive_member=AACT_ARCHIVE_MEMBER,
            order_by=AACT_ORDER_BY,
            scratch_root=self.context.config.work_path,
        )
        additional = [tables['study_references'], tables['brief_summaries'], tables['detailed_descriptions']]

        report = extract_clinical_report(
            studies=tables['studies'].select('nct_id', 'study_type', 'phase', 'official_title'),
            interventions=tables['interventions'],
            conditions=tables['conditions'],
            additional_metadata=additional,
            aggregation_specs={'pmid': {'group_by': 'nct_id', 'alias': 'literature'}},
        )
        return sample_report(report.df, self.spec.sample_size)

    def _build_prompts(self, report: pl.DataFrame, schema_digest: str) -> pl.DataFrame:
        """Render one prompt per trial and derive its cache key."""
        publications = fetch_publications(
            report,
            max_publications=self.spec.publications.max_publications,
            enabled=self.spec.publications.enabled,
        )
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
                cache_key=pl.col('prompt_sha256').map_elements(
                    lambda d: cache_key(d, schema_digest), return_dtype=pl.String
                ),
            )
        )

    def _extract(self, shard: pl.DataFrame, system_prompt_path: str) -> pl.DataFrame:
        """Run the LLM over one shard and return a cache row per trial it extracted.

        ``run_extraction`` returns only what succeeded and logs the rest, so
        trials the model failed on are simply absent from the result, and
        therefore absent from the cache. The next run tries them again. Nothing
        records that they failed — see :py:func:`pts.result_cache.cached_map`.
        """
        prompts = shard.select('id', 'prompt').to_dicts()
        if len(prompts) == 1:
            # run_extraction treats a single prompt as interactive inspect mode
            # and returns None, so never hand it exactly one
            prompts = prompts * 2

        extracted = run_extraction(
            prompts=prompts,
            model_class=self.spec.model_class,
            system_prompt_path=system_prompt_path,
            model=self.spec.model,
            openai_key=Path(self.spec.openai_key_path).read_text(encoding='utf-8').strip(),
            service_tier=self.spec.service_tier,
            concurrency=self.spec.concurrency,
            max_retries=self.spec.max_retries,
        )
        if extracted is None or extracted.is_empty():
            extracted = pl.DataFrame(schema={'id': pl.String})

        keys = shard.select('id', 'cache_key', 'prompt_sha256')
        succeeded = extracted.unique(subset='id').join(keys, on='id', how='inner')

        missing = shard.height - succeeded.height
        if missing:
            logger.warning(f'{missing} of {shard.height} trials in this shard returned no extraction')

        return succeeded

    @report
    def run(self) -> Self:
        system_prompt = self._system_prompt()
        model_cls = _import_class(self.spec.model_class)
        schema_digest = hashlib.sha256(
            json.dumps(model_cls.model_json_schema(by_alias=True), sort_keys=True).encode('utf-8')
        ).hexdigest()

        report = self._trial_report()
        prompts = self._build_prompts(report, schema_digest)
        logger.info(f'built {prompts.height} prompts from {report.height} trials')

        if self.spec.legacy_batch_results:
            imported = self._seed_legacy_cache(prompts)
            if self.spec.legacy_import_only:
                self.stats = {
                    'trials': report.height,
                    'prompts': prompts.height,
                    'extracted': imported,
                    'failed': prompts.height - imported,
                }
                logger.info(f'legacy cache import complete: {self.stats}')
                return self

        # run_extraction reads the system prompt off disk, so give it a local copy
        system_prompt_path = f'{self.context.config.work_path}/.scratch/system_prompt.txt'
        StorageHandle(system_prompt_path, config=self.context.config).write_text(system_prompt)

        extractions = cached_map(
            records=prompts,
            compute=lambda shard: self._extract(shard, system_prompt_path),
            cache_uri=self.spec.cache_uri,
            config=self.context.config,
            run_id=self.spec.snapshot,
            timestamp=self.spec.snapshot,
            shard_size=self.spec.shard_size,
        )

        published = extractions.drop('computed_at')
        self.stats = {
            'trials': report.height,
            'prompts': prompts.height,
            'extracted': published.height,
            'failed': prompts.height - published.height,
        }
        logger.info(f'extraction complete: {self.stats}')

        self._publish(prompts, published)
        return self

    def _seed_legacy_cache(self, prompts: pl.DataFrame) -> int:
        """Import legacy Batch API results using the current prompt-derived keys.

        This compatibility path is deliberately opt-in and intended for one
        migration run. Failed and malformed legacy records remain absent from
        the cache, so the normal extraction loop retries them.
        """
        legacy_path = self.spec.legacy_batch_results
        assert legacy_path is not None
        logger.info(f'importing legacy AACT batch results from {legacy_path}')
        legacy = parse_batch_results(legacy_path).df
        if legacy.is_empty():
            logger.warning('legacy batch results contained no valid extractions')
            return 0

        # A trial may occur in more than one historical batch. Keep the last
        # parsed occurrence deterministically; the cache has one value per key.
        legacy = legacy.unique(subset='id', keep='last')
        imported = legacy.join(prompts.select('id', 'cache_key', 'prompt_sha256'), on='id', how='inner').unique(
            subset='cache_key', keep='last'
        )
        if imported.is_empty():
            logger.warning('no legacy extractions matched the current AACT prompts')
            return 0

        existing = read_cache(self.spec.cache_uri, self.context.config)
        combined = pl.concat([existing, imported], how='diagonal_relaxed')
        combined = combined.with_columns(
            computed_at=pl.coalesce([
                pl.col('computed_at'),
                pl.lit(datetime.now(UTC)).cast(METADATA_SCHEMA['computed_at']),
            ])
        )
        combined = combined.sort('computed_at', descending=True).unique(subset='cache_key', keep='first')
        write_cache(combined, self.spec.cache_uri, self.context.config, self.spec.snapshot)
        logger.info(f'seeded {imported.height} legacy extractions into the cache')
        return imported.height

    def _publish(self, prompts: pl.DataFrame, extractions: pl.DataFrame) -> None:
        """Write the prompts and the extractions."""
        artifacts = []
        for key, df in (('prompts', prompts), ('extraction', extractions)):
            handle = StorageHandle(self.spec.destination[key], config=self.context.config)
            df.write_parquet(handle.absolute, compression='zstd')
            artifacts.append(Artifact(source=self.spec.cache_uri, destination=handle.absolute))

        self.artifacts = artifacts

    @report
    def validate(self) -> Self:
        """Check the published extraction covers the trials it was asked about.

        A trial the model answered negatively still has a row, so a missing row
        means the trial was never successfully attempted. That is a real gap,
        but it is not a reason to stop a release: some trials fail permanently,
        and the release is better off with a slightly thinner extraction than
        with no release. Hence a warning rather than an error.
        """
        if not self.stats:
            raise TaskValidationError('run produced no statistics')

        prompts = self.stats['prompts']
        extracted = self.stats['extracted']
        if extracted < prompts:
            missing = prompts - extracted
            logger.warning(
                f'{missing} of {prompts} trials ({missing / prompts:.1%}) have no extraction. '
                f'a negative result is still a row, so these were never successfully attempted'
            )
        return self


def _import_class(dotted_path: str) -> type[BaseModel]:
    """Import a pydantic model class from a dotted module path."""
    module_path, class_name = dotted_path.rsplit('.', 1)
    return getattr(import_module(module_path), class_name)
