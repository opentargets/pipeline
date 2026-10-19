"""Task that extracts structured drug and disease evidence from clinical trials with an LLM.

The extraction itself lives in :py:mod:`pts.transformers.clinical`; this task is the part that
makes it repeatable — it builds one prompt per trial, asks
:py:mod:`pts.result_cache` for the ones that have not been extracted yet, and
publishes the result as a typed parquet the release can read directly.

Cache identity is the trial ID and the JSON schema the model is held to. The
prompt hash is retained as audit metadata, but prompt text, the
model and system instructions do not invalidate an accepted extraction.

The extraction work lives in :py:mod:`pts.aact_trial_extraction`.
"""

from __future__ import annotations

import hashlib
import json
from importlib import import_module, resources
from pathlib import Path
from typing import Any, Self

import polars as pl
from loguru import logger
from otter.manifest.model import Artifact
from otter.storage.synchronous.handle import StorageHandle
from otter.task.model import Spec, Task, TaskContext
from otter.task.task_reporter import report
from otter.util.errors import TaskValidationError
from pydantic import BaseModel

from pts.result_cache import cached_map

DEFAULT_SYSTEM_PROMPT = ('pts.transformers.clinical.prompts', 'aact_llm.txt')
"""Packaged system prompt."""


def _schema_cache_uri(cache_uri: str, schema_digest: str) -> str:
    """Return the cache namespace for one response schema."""
    return f'{cache_uri.rstrip("/")}/{schema_digest}'


class LlmExtractSpec(Spec):
    """Configuration fields for the llm_extract task."""

    source: dict[str, str]
    """AACT dump archive. The task restores the required tables inline."""
    destination: dict[str, str]
    """Where to publish. Keys: ``prompts``, ``extraction``, ``errors``."""
    cache_uri: str
    """Root containing one extraction cache per response schema, an absolute
        URI. Must sit outside the dataset version directory so a trial whose
        text did not change between two AACT snapshots stays a cache hit."""
    snapshot: str
    """Name for the cache snapshot this run writes, and for its staging area.
        Reuse it to resume a failed run; change it to start clean. The DAG sets
        it from the AACT version."""
    openai_key_path: str = '/var/run/secrets/openai-api-key'
    """Path the OpenAI key is mounted at, injected from Secret Manager by the
        DAG. Never put the key itself in config."""
    model: str = 'gpt-5-nano-2025-08-07'
    """OpenAI model. Not part of the cache key."""
    model_class: str = 'pts.transformers.clinical.schemas.ClinicalReportExtractionSchema'
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
    legacy_batch_results: str | None = None
    """Optional one-off directory of legacy OpenAI Batch API results to seed the cache,
        relative to the release root."""
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
        self.errors: list[str] = []

    def _system_prompt(self) -> str:
        """Return the system prompt text, from config if overridden and from the package otherwise."""
        if self.spec.system_prompt:
            text, _ = StorageHandle(self.spec.system_prompt, config=self.context.config).read_text()
            return text
        package, filename = DEFAULT_SYSTEM_PROMPT
        return (resources.files(package) / filename).read_text(encoding='utf-8')

    @report
    def run(self) -> Self:
        from pts import aact_trial_extraction

        system_prompt = self._system_prompt()
        model_cls = _import_class(self.spec.model_class)
        schema_digest = hashlib.sha256(
            json.dumps(model_cls.model_json_schema(by_alias=True), sort_keys=True).encode('utf-8')
        ).hexdigest()
        schema_cache_uri = _schema_cache_uri(self.spec.cache_uri, schema_digest)

        archive = StorageHandle(str(self.spec.source['aact']), config=self.context.config).absolute
        report = aact_trial_extraction.trial_report(archive, self.context.config.work_path, self.spec.sample_size)
        prompts = aact_trial_extraction.prompts_for(report, schema_digest)
        logger.info(f'built {prompts.height} prompts from {report.height} trials')

        if self.spec.legacy_batch_results:
            imported = aact_trial_extraction.seed_legacy_cache(
                prompts,
                self.spec.legacy_batch_results,
                schema_cache_uri,
                self.context.config,
                self.spec.snapshot,
            )
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
        Path(system_prompt_path).parent.mkdir(parents=True, exist_ok=True)
        StorageHandle(system_prompt_path, config=self.context.config, force_local=True).write_text(system_prompt)

        extractions = cached_map(
            records=prompts,
            compute=lambda shard: aact_trial_extraction.extract(shard, self.spec, system_prompt_path, self.errors),
            cache_uri=schema_cache_uri,
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

        self._publish(prompts, published, schema_cache_uri)
        return self

    def _publish(self, prompts: pl.DataFrame, extractions: pl.DataFrame, schema_cache_uri: str) -> None:
        """Write the prompts, extractions and diagnostics."""
        artifacts = []
        for key, df in (('prompts', prompts), ('extraction', extractions)):
            handle = StorageHandle(self.spec.destination[key], config=self.context.config)
            df.write_parquet(handle.absolute, compression='zstd')
            artifacts.append(Artifact(source=schema_cache_uri, destination=handle.absolute))

        errors = StorageHandle(self.spec.destination['errors'], config=self.context.config)
        errors.write_text(''.join(f'{error}\n' for error in self.errors))
        artifacts.append(Artifact(source=schema_cache_uri, destination=errors.absolute))
        logger.info(f'wrote {len(self.errors)} extraction diagnostics to {errors.absolute}')

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
