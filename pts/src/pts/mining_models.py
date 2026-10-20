"""Implementation of the :py:mod:`pts.tasks.mining_models` task."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars as pl

from pts.aact_trial_models import tracked
from pts.record_store import map_texts, publish_snapshot
from pts.transformers.clinical.hf_models import offline_hf_models, revision_tags, staged_model_revisions
from pts.transformers.clinical.mining import RecordingNer, build_mining_reports, report_drug_labels
from pts.transformers.clinical.ner import (
    DISEASE_NER_MODEL,
    DRUG_NER_MODEL,
    disease_entities_with_status,
    drug_entities_with_status,
    drug_ner_revision,
)

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
    max_attempts: int,
) -> dict[str, Any]:
    """Run disease NER over the EMA and PMDA indication texts and drug NER over the unresolved drug labels.

    Returns:
        dict[str, Any]: Per model, the number of texts and whether it computed or published anything.
    """
    revisions = staged_model_revisions(models_uri)
    tags = revision_tags(revisions)
    recorder = RecordingNer()
    reports = build_mining_reports(ttd, ema, pmda, recorder)
    disease_texts = pl.Series('text', sorted(recorder.texts), dtype=pl.String)
    drug_labels = report_drug_labels(report.df for report in reports.values())
    disease_calls: list[int] = []
    ner_calls: list[int] = []
    ner_revision = drug_ner_revision(revisions)

    with offline_hf_models(models_uri):
        disease = map_texts(
            disease_texts,
            tracked(disease_entities_with_status, disease_calls),
            f'{store_uri}/disease_entities',
            DISEASE_NER_MODEL,
            revisions[DISEASE_NER_MODEL],
            run_id,
            config,
            max_attempts,
        )
        ner = map_texts(
            drug_labels,
            tracked(lambda texts: drug_entities_with_status(texts, ner_batch_size), ner_calls),
            f'{store_uri}/drug_entities',
            DRUG_NER_MODEL,
            ner_revision,
            run_id,
            config,
            max_attempts,
        )

    disease_published = publish_snapshot(
        disease.select(text=pl.col('text'), extracted_diseases=pl.col('output')),
        f'{destination["disease_entities"]}/{tags["disease_entities"]}',
        {
            'model': DISEASE_NER_MODEL,
            'model_revision': revisions[DISEASE_NER_MODEL],
            'model_revisions': revisions,
        },
        config,
    )
    ner_published = publish_snapshot(
        ner.select(label=pl.col('text'), extracted_drugs=pl.col('output')),
        f'{destination["drug_entities"]}/{tags["drug_entities"]}',
        {'model': DRUG_NER_MODEL, 'model_revision': ner_revision, 'model_revisions': revisions},
        config,
    )
    return {
        'disease_entities': {'texts': disease.height, 'computed': bool(disease_calls), 'published': disease_published},
        'drug_entities': {'texts': ner.height, 'computed': bool(ner_calls), 'published': ner_published},
    }
