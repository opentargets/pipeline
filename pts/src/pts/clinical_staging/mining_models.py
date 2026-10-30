"""Implementation of the :py:mod:`pts.tasks.mining_models` task."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars as pl

from pts.clinical_staging.staged_outputs import DISEASE_ENTITIES, DRUG_ENTITIES
from pts.transformers.clinical.hf_models import offline_hf_models, staged_model_revisions
from pts.transformers.clinical.mining import RecordingNer, build_mining_reports, report_drug_labels, staged_lookup
from pts.transformers.clinical.ner import disease_entities_with_status, drug_entities_with_status

if TYPE_CHECKING:
    from otter.config.model import Config


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

    Returns:
        dict[str, Any]: Per model, the number of texts and whether it computed or published anything.
    """
    revisions = staged_model_revisions(models_uri)
    DISEASE_ENTITIES.require(revisions)
    DRUG_ENTITIES.require(revisions)
    recorder = RecordingNer()
    build_mining_reports(ttd, ema, pmda, recorder)
    disease_texts = pl.Series('text', sorted(recorder.texts), dtype=pl.String)

    with offline_hf_models(models_uri):
        disease = DISEASE_ENTITIES.stage(
            disease_texts,
            lambda texts: disease_entities_with_status(texts, ner_batch_size, ner_model_batch_size),
            revisions,
            store_uri,
            destination['disease_entities'],
            run_id,
            config,
            max_attempts,
        )
        reports = build_mining_reports(
            ttd, ema, pmda, staged_lookup(disease.frame, 'text', 'extracted_diseases', 'disease texts')
        )
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
