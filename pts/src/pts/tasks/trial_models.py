"""Task that runs the stop reason classifier and drug NER over AACT texts, once per AACT version.

The models run through the record store of :py:mod:`pts.record_store`, so a text
that already has an answer for the model revision is not computed again, and the
result is published as a write-once snapshot next to the AACT version.

The work lives in :py:mod:`pts.clinical_staging.trial_models`.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Self

from loguru import logger
from otter.manifest.model import Artifact
from otter.storage.synchronous.handle import StorageHandle
from otter.task.model import Spec, Task, TaskContext
from otter.task.task_reporter import report
from otter.util.errors import TaskValidationError


class TrialModelsSpec(Spec):
    """Configuration fields for the trial_models task."""

    source: dict[str, str]
    """Inputs. Keys: ``aact`` (the dump archive), ``extraction`` (the extraction
        parquet), ``models`` (the staged Hugging Face models)."""
    destination: dict[str, str]
    """Where to publish the snapshots. Keys: ``stop_reasons``, ``drug_entities``."""
    store_uri: str
    """Root of the record stores, an absolute URI. Sits outside the AACT version
        directory so a text seen in an earlier version is not computed again."""
    snapshot: str
    """The AACT version, which names the records this run writes."""
    ner_batch_size: int = 256
    """Labels per drug NER batch."""
    stop_reasons_batch_size: int = 32
    """Texts per stop reasons batch."""
    max_attempts: int = 3
    """Failed records after which a text is no longer retried."""
    seed_ner_cache: str | None = None
    """Optional one-off release NER cache (parquet) to import into the drug entities store."""


class TrialModels(Task):
    """Stage stop reasons and drug entities for the texts of one AACT version."""

    def __init__(self, spec: TrialModelsSpec, context: TaskContext) -> None:
        super().__init__(spec, context)
        self.spec: TrialModelsSpec
        self.stats: dict[str, Any] = {}

    @report
    def run(self) -> Self:
        import io

        import polars as pl

        from pts.clinical_staging import trial_models
        from pts.clinical_staging.staged_outputs import DRUG_ENTITIES
        from pts.postgres import read_dump_tables
        from pts.transformers.clinical.hf_models import staged_model_revisions
        from pts.transformers.clinical.providers.aact import AACT_ARCHIVE_MEMBER, AACT_SCHEMA_NAME

        config = self.context.config
        models_uri = StorageHandle(self.spec.source['models'], config=config).absolute
        store_uri = self.spec.store_uri.rstrip('/')

        if self.spec.seed_ner_cache:
            revision = DRUG_ENTITIES.tag(staged_model_revisions(models_uri))
            seeded = trial_models.seed_drug_entities(
                StorageHandle(self.spec.seed_ner_cache, config=config).absolute,
                store_uri,
                DRUG_ENTITIES.model,
                revision,
                config,
            )
            logger.info(f'seeded {seeded} drug entity records')

        archive = StorageHandle(self.spec.source['aact'], config=config).absolute
        studies = read_dump_tables(
            archive,
            {'studies': ['nct_id', 'why_stopped']},
            schema_name=AACT_SCHEMA_NAME,
            archive_member=AACT_ARCHIVE_MEMBER,
            scratch_root=config.work_path,
        )['studies']
        data, _ = StorageHandle(self.spec.source['extraction'], config=config).read()
        extractions = pl.read_parquet(io.BytesIO(data))

        run_id = f'{self.spec.snapshot}-{datetime.now(UTC):%Y%m%dT%H%M%SZ}'
        destination = {key: StorageHandle(uri, config=config).absolute for key, uri in self.spec.destination.items()}
        self.stats = trial_models.stage_trial_models(
            studies,
            extractions,
            models_uri,
            destination,
            store_uri,
            run_id,
            config,
            ner_batch_size=self.spec.ner_batch_size,
            stop_reasons_batch_size=self.spec.stop_reasons_batch_size,
            max_attempts=self.spec.max_attempts,
        )
        logger.info(f'trial models staged: {self.stats}')

        self.artifacts = [Artifact(source=store_uri, destination=uri) for uri in destination.values()]
        return self

    @report
    def validate(self) -> Self:
        """Check the run produced statistics for both models."""
        if not self.stats:
            raise TaskValidationError('run produced no statistics')
        return self
