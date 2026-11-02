"""Transformer that runs disease and drug NER over the TTD, EMA and PMDA texts, once per clinical_mining version.

The models run through the record store of :py:mod:`pts.transformers.clinical.record_store`, so a text
that already has an answer for the model revision is not computed again, and the
result is published as a write-once snapshot under the model revision tag.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars as pl
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from pts.transformers.clinical.hf_models import offline_hf_models, staged_model_revisions
from pts.transformers.clinical.mining import (
    RecordingNer,
    mining_reports,
    read_mining_inputs,
    report_drug_labels,
    staged_lookup,
)
from pts.transformers.clinical.ner import disease_entities_with_status, drug_entities_with_status
from pts.transformers.clinical.record_store import new_run_id
from pts.transformers.clinical.staged_outputs import DISEASE_ENTITIES, DRUG_ENTITIES

if TYPE_CHECKING:
    from otter.config.model import Config


class ClinicalMiningModelsSettings(BaseModel):
    """Settings for the clinical_mining_models transformer."""

    model_config = ConfigDict(extra='forbid')

    store_uri: str
    """Root of the record stores, an absolute URI. Shared with the AACT staging, so a
        label seen there is not computed again."""
    snapshot: str
    """The clinical_mining version, which names the records this run writes."""
    ner_batch_size: int = Field(default=256, gt=0)
    """Texts per disease and drug NER batch."""
    ner_model_batch_size: int = Field(default=32, gt=0)
    """Labels per drug NER forward pass, within a batch. The disease NER runs one text at a time."""
    max_attempts: int = 3
    """Failed records after which a text is no longer retried."""


def stage_mining_models(
    ttd: str,
    ema: str,
    pmda: str,
    models_uri: str,
    destination: dict[str, str],
    store_uri: str,
    run_id: str,
    config: Config,
    ner_batch_size: int,
    ner_model_batch_size: int,
    max_attempts: int,
) -> dict[str, Any]:
    """Run disease NER over the EMA and PMDA indication texts and drug NER over the unresolved drug labels.

    The inputs are read once. The reports are built twice: first with a recording NER, to collect the disease
    texts, then with the staged disease results. The second build is needed for the drug labels, because a PMDA
    row whose notes yield no disease is dropped, and its active ingredients with it.

    Returns:
        dict[str, Any]: Per model, the number of texts and whether it computed or published anything.
    """
    revisions = staged_model_revisions(models_uri)
    DISEASE_ENTITIES.require(revisions)
    DRUG_ENTITIES.require(revisions)
    inputs = read_mining_inputs(ttd, ema, pmda)
    recorder = RecordingNer()
    mining_reports(inputs, recorder)
    disease_texts = pl.Series('text', sorted(recorder.texts), dtype=pl.String)

    with offline_hf_models(models_uri):
        disease = DISEASE_ENTITIES.stage(
            disease_texts,
            lambda texts: disease_entities_with_status(texts, ner_batch_size),
            revisions,
            store_uri,
            destination['disease_entities'],
            run_id,
            config,
            max_attempts,
        )
        reports = mining_reports(inputs, staged_lookup(disease.frame, 'text', 'extracted_diseases', 'disease texts'))
        drug_labels = report_drug_labels(report.df for report in reports.values())
        ner = DRUG_ENTITIES.stage(
            drug_labels,
            lambda texts: drug_entities_with_status(texts, ner_batch_size, ner_model_batch_size),
            revisions,
            store_uri,
            destination['drug_entities'],
            run_id,
            config,
            max_attempts,
        )

    return {'disease_entities': disease.stats(), 'drug_entities': ner.stats()}


def clinical_mining_models(
    source: dict[str, str], destination: dict[str, str], settings: dict[str, Any], config: Config
) -> None:
    """Stage disease and drug entities for the texts of one clinical_mining version.

    Args:
        source: Keys ``ttd``, ``ema``, ``pmda`` (the provider files) and ``models`` (the staged Hugging Face models).
        destination: Where to publish the snapshots. Keys ``drug_entities`` and ``disease_entities``.
        settings: See :py:class:`ClinicalMiningModelsSettings`.
        config: The otter config.
    """
    s = ClinicalMiningModelsSettings.model_validate(settings)
    store_uri = s.store_uri.rstrip('/')
    run_id = new_run_id(s.snapshot)
    stats = stage_mining_models(
        source['ttd'],
        source['ema'],
        source['pmda'],
        source['models'],
        destination,
        store_uri,
        run_id,
        config,
        ner_batch_size=s.ner_batch_size,
        ner_model_batch_size=s.ner_model_batch_size,
        max_attempts=s.max_attempts,
    )
    logger.info(f'mining models staged: {stats}')
