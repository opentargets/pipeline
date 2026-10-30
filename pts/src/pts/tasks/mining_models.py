"""Task that runs disease and drug NER over the TTD, EMA and PMDA texts, once per clinical_mining version.

The models run through the record store of :py:mod:`pts.record_store`, so a text
that already has an answer for the model revision is not computed again, and the
result is published as a write-once snapshot under the model revision tag.

The work lives in :py:mod:`pts.clinical_staging.mining_models`.
"""

from __future__ import annotations

from typing import Any, Self

from loguru import logger
from otter.manifest.model import Artifact
from otter.storage.synchronous.handle import StorageHandle
from otter.task.model import Spec, Task, TaskContext
from otter.task.task_reporter import report
from otter.util.errors import TaskValidationError


class MiningModelsSpec(Spec):
    """Configuration fields for the mining_models task."""

    source: dict[str, str]
    """Inputs. Keys: ``ttd``, ``ema``, ``pmda`` (the provider files), ``models`` (the staged Hugging Face models)."""
    destination: dict[str, str]
    """Where to publish the snapshots. Keys: ``drug_entities``, ``disease_entities``."""
    store_uri: str
    """Root of the record stores, an absolute URI. Shared with the AACT staging, so a
        label seen there is not computed again."""
    snapshot: str
    """The clinical_mining version, which names the records this run writes."""
    ner_batch_size: int = 256
    """Labels per drug NER batch."""
    max_attempts: int = 3
    """Failed records after which a text is no longer retried."""


class MiningModels(Task):
    """Stage disease and drug entities for the texts of one clinical_mining version."""

    def __init__(self, spec: MiningModelsSpec, context: TaskContext) -> None:
        super().__init__(spec, context)
        self.spec: MiningModelsSpec
        self.stats: dict[str, Any] = {}

    @report
    def run(self) -> Self:
        from pts.clinical_staging import mining_models
        from pts.record_store import new_run_id

        config = self.context.config
        source = {key: StorageHandle(uri, config=config).absolute for key, uri in self.spec.source.items()}
        store_uri = self.spec.store_uri.rstrip('/')
        run_id = new_run_id(self.spec.snapshot)
        destination = {key: StorageHandle(uri, config=config).absolute for key, uri in self.spec.destination.items()}
        self.stats = mining_models.stage_mining_models(
            source['ttd'],
            source['ema'],
            source['pmda'],
            source['models'],
            destination,
            store_uri,
            run_id,
            config,
            ner_batch_size=self.spec.ner_batch_size,
            max_attempts=self.spec.max_attempts,
        )
        logger.info(f'mining models staged: {self.stats}')

        self.artifacts = [Artifact(source=store_uri, destination=uri) for uri in destination.values()]
        return self

    @report
    def validate(self) -> Self:
        """Check the run produced statistics for both models."""
        if not self.stats:
            raise TaskValidationError('run produced no statistics')
        return self
