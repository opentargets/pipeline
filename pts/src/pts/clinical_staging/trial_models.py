"""Implementation of the :py:mod:`pts.tasks.trial_models` task."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import polars as pl
from loguru import logger

from pts.clinical_staging.staged_outputs import DRUG_ENTITIES, STOP_REASONS
from pts.record_store import RECORD_SCHEMA, read_parquet, run_exists, text_sha256, write_run
from pts.transformers.clinical.helpers import sanitise_text
from pts.transformers.clinical.hf_models import offline_hf_models, staged_model_revisions
from pts.transformers.clinical.ner import drug_entities_with_status
from pts.transformers.clinical.providers.aact import llm_drug_labels
from pts.transformers.clinical.stop_reasons import stop_reasons_with_status

if TYPE_CHECKING:
    from otter.config.model import Config

SEED_RUN_ID = 'seed-ner-cache'
SEED_COMPUTED_AT = datetime(1970, 1, 1, tzinfo=UTC)
"""Run that holds the records imported from the release NER cache."""


def why_stopped_texts(studies: pl.DataFrame) -> pl.Series:
    """Distinct non-null ``why_stopped`` texts, sorted, as the release's ``ClinicalReport`` sanitises them."""
    sanitised = studies['why_stopped'].map_elements(sanitise_text, return_dtype=pl.String, skip_nulls=True)
    return sanitised.drop_nulls().unique().sort()


def seed_drug_entities(ner_cache_uri: str, store_uri: str, model: str, model_revision: str, config: Config) -> int:
    """Import a release NER cache into the drug entities store, once.

    Returns:
        int: The number of records written, 0 if the seed run already exists.
    """
    drug_store = f'{store_uri}/drug_entities'
    if run_exists(drug_store, SEED_RUN_ID, config):
        logger.info(f'{SEED_RUN_ID} already imported into {drug_store}')
        return 0

    cache = (
        read_parquet(ner_cache_uri, config)
        .select('query_label', 'extracted_drugs')
        .drop_nulls('query_label')
        .unique(subset='query_label', keep='first', maintain_order=True)
    )
    records = cache.select(
        text_sha256=pl.col('query_label').map_elements(text_sha256, return_dtype=pl.String),
        text=pl.col('query_label'),
        model=pl.lit(model),
        model_revision=pl.lit(model_revision),
        output=pl.col('extracted_drugs').fill_null([]),
        status=pl.lit('ok'),
        error=pl.lit(None, dtype=pl.String),
        computed_at=pl.lit(SEED_COMPUTED_AT),
        run_id=pl.lit(SEED_RUN_ID),
    )
    write_run(records.cast(RECORD_SCHEMA), drug_store, SEED_RUN_ID, config)
    return records.height


def stage_trial_models(
    studies: pl.DataFrame,
    extractions: pl.DataFrame,
    models_uri: str,
    destination: dict[str, str],
    store_uri: str,
    run_id: str,
    config: Config,
    ner_batch_size: int,
    stop_reasons_batch_size: int,
    max_attempts: int,
) -> dict[str, Any]:
    """Run the stop reason classifier and drug NER over the AACT texts, and publish both snapshots.

    Returns:
        dict[str, Any]: Per model, the number of texts and whether it computed or published anything.
    """
    revisions = staged_model_revisions(models_uri)
    STOP_REASONS.require(revisions)
    DRUG_ENTITIES.require(revisions)
    stop_texts = why_stopped_texts(studies)
    drug_texts = llm_drug_labels(extractions)

    with offline_hf_models(models_uri):
        stop = STOP_REASONS.stage(
            stop_texts,
            lambda texts: stop_reasons_with_status(texts, stop_reasons_batch_size),
            revisions,
            store_uri,
            destination['stop_reasons'],
            run_id,
            config,
            max_attempts,
        )
        ner = DRUG_ENTITIES.stage(
            drug_texts,
            lambda texts: drug_entities_with_status(texts, ner_batch_size),
            revisions,
            store_uri,
            destination['drug_entities'],
            run_id,
            config,
            max_attempts,
        )

    return {
        name: {
            'texts': result.frame.height,
            'computed': result.computed,
            'published': result.published,
            'tag': result.tag,
        }
        for name, result in (('stop_reasons', stop), ('drug_entities', ner))
    }
