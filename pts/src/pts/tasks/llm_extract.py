"""Task that extracts structured drug and disease evidence from clinical trials with an LLM.

The extraction itself lives in :py:mod:`pts.transformers.clinical`; this task is the part that
makes it repeatable. It builds one prompt per trial, asks the
:py:mod:`pts.record_store` store for the ones not yet extracted under the current
response schema, and publishes the extraction as a write-once snapshot the
release reads directly.

A record's text is the rendered prompt, so a trial whose text changed is
extracted again. The model and system instructions are not part of the record's
identity, so changing them re-extracts nothing. A trial that fails every attempt
of a run is left out of the snapshot, and a rerun of the same AACT version does
not try it again.

The extraction work lives in :py:mod:`pts.clinical_staging.extraction`.
"""

from __future__ import annotations

from importlib import resources
from typing import Any, Self

import polars as pl
from loguru import logger
from otter.manifest.model import Artifact
from otter.storage.synchronous.handle import StorageHandle
from otter.task.model import Spec, Task, TaskContext
from otter.task.task_reporter import report
from otter.util.errors import TaskRunError, TaskValidationError

from pts.record_store import map_texts, new_run_id, publish_snapshot

DEFAULT_SYSTEM_PROMPT = ('pts.transformers.clinical.prompts', 'aact_llm.txt')
"""Packaged system prompt."""


class LlmExtractSpec(Spec):
    """Configuration fields for the llm_extract task."""

    source: dict[str, str]
    """AACT dump archive. The task restores the required tables inline."""
    destination: dict[str, str]
    """Where to publish. Key: ``extraction``, the directory holding one snapshot per tag."""
    store_uri: str
    """Root of the record stores, an absolute URI. Must sit outside the dataset
        version directory so a trial whose prompt did not change between two AACT
        snapshots is not extracted again."""
    snapshot: str
    """The AACT version. Prefixes the id of the runs this task writes to the store."""
    openai_key_path: str = '/var/run/secrets/openai-api-key'
    """Path the OpenAI key is mounted at, injected from Secret Manager by the
        DAG. Never put the key itself in config."""
    model: str = 'gpt-5-nano-2025-08-07'
    """OpenAI model. Not part of a record's identity."""
    model_class: str = 'pts.transformers.clinical.schemas.ClinicalReportExtractionSchema'
    """Dotted path to the pydantic model the response is validated against. The
        digest of its JSON schema is the records' model revision."""
    system_prompt: str | None = None
    """Path to a system prompt overriding the packaged one, relative to the
        release root. Its contents are not part of a record's identity."""
    service_tier: str = 'flex'
    """OpenAI service tier. ``flex`` is roughly half price for higher latency,
        with an automatic fallback to standard when flex capacity runs out."""
    concurrency: int = 50
    """Parallel API calls in flight."""
    max_retries: int = 2
    """SDK-level retries for transient errors."""
    shard_size: int = 2000
    """Trials per shard. Each shard is written to the store as it finishes, so a
        retry of this task resumes rather than paying for the same calls twice."""
    max_attempts: int = 3
    """Attempts per trial, within a run and across runs, before it is left out."""
    sample_size: int | None = None
    """Extract a random sample only. For dry runs; leave unset in production."""
    import_extraction_cache: str | None = None
    """Extraction cache namespace of the current response schema to import into
        the store, once, before extracting. For the migration off the cache."""


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

    @report
    def run(self) -> Self:
        from pts.clinical_staging import extraction
        from pts.clinical_staging.staged_outputs import EXTRACTION, schema_digest

        system_prompt = self._system_prompt()
        model_cls = extraction.import_class(self.spec.model_class)
        digest = schema_digest(model_cls)
        extraction.polars_schema(model_cls)
        config = self.context.config
        store_uri = f'{self.spec.store_uri.rstrip("/")}/{EXTRACTION.name}'

        archive = StorageHandle(str(self.spec.source['aact']), config=config).absolute
        report = extraction.trial_report(archive, config.work_path, self.spec.sample_size)
        prompts = extraction.prompts_for(report)
        logger.info(f'built {prompts.height} prompts from {report.height} trials')

        if self.spec.import_extraction_cache:
            imported = extraction.import_extraction_cache(
                self.spec.import_extraction_cache, prompts, store_uri, digest, config
            )
            logger.info(f'imported {imported} cached extractions into {store_uri}')

        mapped = map_texts(
            prompts['prompt'],
            extraction.extraction_compute(prompts, self.spec, system_prompt),
            store_uri,
            extraction.EXTRACTION_MODEL,
            digest,
            new_run_id(self.spec.snapshot),
            config,
            max_attempts=self.spec.max_attempts,
            output_dtype=pl.String(),
            allow_failures=True,
            shard_size=self.spec.shard_size,
        )
        extractions = extraction.extractions_from_records(mapped.frame, prompts, model_cls)
        if prompts.height and extractions.is_empty():
            msg = f'no trial of {prompts.height} was extracted, so no snapshot is published'
            raise TaskRunError(msg)

        destination = StorageHandle(self.spec.destination['extraction'], config=config).absolute
        tag = EXTRACTION.tag(digest, self.spec.sample_size)
        published = publish_snapshot(
            extractions,
            f'{destination}/{tag}',
            EXTRACTION.metadata(digest, self.spec.sample_size),
            config,
        )
        self.artifacts = [Artifact(source=store_uri, destination=f'{destination}/{tag}')]
        self.stats = {
            'trials': report.height,
            'prompts': prompts.height,
            'extracted': extractions.height,
            'computed': mapped.computed,
            'failed': mapped.failed,
            'published': published,
        }
        logger.info(f'extraction complete: {self.stats}')
        return self

    @report
    def validate(self) -> Self:
        """Check the published extraction covers the trials it was asked about.

        A trial the model answered negatively still has a row, so a missing row
        means the trial was never successfully attempted. Missing rows are
        logged as a warning.
        """
        if not self.stats:
            raise TaskValidationError('run produced no statistics')

        prompts = self.stats['prompts']
        extracted = self.stats['extracted']
        if extracted < prompts:
            missing = prompts - extracted
            logger.warning(
                f'{missing} of {prompts} trials ({missing / prompts:.1%}) have no extraction. '
                f'a negative result is still a row, so these failed all of their attempts'
            )
        return self
