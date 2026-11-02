"""Transformer that runs the stop reason classifier and drug NER over AACT texts, once per AACT version.

The models run through the record store of :py:mod:`pts.transformers.clinical.record_store`, so a text
that already has an answer for the model revision is not computed again, and the
result is published as a write-once snapshot next to the AACT version.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import polars as pl
from loguru import logger
from otter.storage.synchronous.handle import StorageHandle
from otter.util.errors import NotFoundError
from pydantic import BaseModel, ConfigDict, Field

from pts.transformers.clinical.helpers import sanitise_text
from pts.transformers.clinical.hf_models import offline_hf_models, staged_model_revisions
from pts.transformers.clinical.ner import drug_entities_with_status
from pts.transformers.clinical.providers.aact import llm_drug_labels, read_aact_tables
from pts.transformers.clinical.record_store import (
    RECORD_SCHEMA,
    import_run,
    new_run_id,
    read_parquet,
    text_sha256,
)
from pts.transformers.clinical.staged_outputs import DRUG_ENTITIES, EXTRACTION, STOP_REASONS
from pts.transformers.clinical.stop_reasons import stop_reasons_with_status

if TYPE_CHECKING:
    from otter.config.model import Config

SEED_RUN_ID = 'seed-ner-cache'
SEED_COMPUTED_AT = datetime(1970, 1, 1, tzinfo=UTC)
"""Run that holds the records imported from the release NER cache."""


class AactTrialModelsSettings(BaseModel):
    """Settings for the aact_trial_models transformer."""

    model_config = ConfigDict(extra='forbid')

    store_uri: str
    """Root of the record stores, an absolute URI. Sits outside the AACT version
        directory so a text seen in an earlier version is not computed again."""
    snapshot: str
    """The AACT version, which names the records this run writes."""
    ner_batch_size: int = Field(default=256, gt=0)
    """Labels per drug NER batch."""
    ner_model_batch_size: int = Field(default=32, gt=0)
    """Labels per drug NER forward pass, within a batch."""
    stop_reasons_batch_size: int = 32
    """Texts per stop reasons batch."""
    max_attempts: int = 3
    """Failed records after which a text is no longer retried."""
    seed_ner_cache: str | None = None
    """Optional one-off release NER cache (parquet) to import into the drug entities store."""


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

    def records() -> pl.DataFrame:
        cache = (
            read_parquet(ner_cache_uri, config)
            .select('query_label', 'extracted_drugs')
            .drop_nulls('query_label')
            .unique(subset='query_label', keep='first', maintain_order=True)
        )
        return cache.select(
            text_sha256=pl.col('query_label').map_elements(text_sha256, return_dtype=pl.String),
            text=pl.col('query_label'),
            model=pl.lit(model),
            model_revision=pl.lit(model_revision),
            output=pl.col('extracted_drugs').fill_null([]),
            status=pl.lit('ok'),
            error=pl.lit(None, dtype=pl.String),
            computed_at=pl.lit(SEED_COMPUTED_AT),
            run_id=pl.lit(SEED_RUN_ID),
        ).cast(RECORD_SCHEMA)

    return import_run(records, drug_store, SEED_RUN_ID, config)


def read_extraction(extraction_uri: str, config: Config) -> pl.DataFrame:
    """Read the extraction of this code's response schema, published under ``extraction_uri``.

    Raises:
        FileNotFoundError: if there is none under the current tag, naming the path it looked for.
    """
    uri = f'{extraction_uri.rstrip("/")}/{EXTRACTION.release_tag()}/data.parquet'
    try:
        return read_parquet(uri, config)
    except NotFoundError as e:
        msg = f'no extraction under the current tag, expected {uri}; run aact_trial_extraction first'
        raise FileNotFoundError(msg) from e


def stage_trial_models(
    studies: pl.DataFrame,
    extractions: pl.DataFrame,
    models_uri: str,
    destination: dict[str, str],
    store_uri: str,
    run_id: str,
    config: Config,
    ner_batch_size: int,
    ner_model_batch_size: int,
    stop_reasons_batch_size: int,
    max_attempts: int,
    extraction_tag: str,
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
            lambda texts: drug_entities_with_status(texts, ner_batch_size, ner_model_batch_size),
            revisions,
            store_uri,
            destination['drug_entities'],
            run_id,
            config,
            max_attempts,
            extra_metadata={'extraction_tag': extraction_tag},
        )

    return {'stop_reasons': stop.stats(), 'drug_entities': ner.stats()}


def aact_trial_models(
    source: dict[str, str], destination: dict[str, str], settings: dict[str, Any], config: Config
) -> None:
    """Stage stop reasons and drug entities for the texts of one AACT version.

    Args:
        source: Keys ``aact`` (the dump archive), ``extraction`` (the directory of tagged extractions) and
            ``models`` (the staged Hugging Face models).
        destination: Key ``stop_reasons``, where to publish that snapshot. The drug entities go under the
            extraction they were staged from.
        settings: See :py:class:`AactTrialModelsSettings`.
        config: The otter config.
    """
    s = AactTrialModelsSettings.model_validate(settings)
    models_uri = source['models']
    store_uri = s.store_uri.rstrip('/')

    if s.seed_ner_cache:
        revision = DRUG_ENTITIES.tag(staged_model_revisions(models_uri))
        seeded = seed_drug_entities(
            StorageHandle(s.seed_ner_cache, config=config).absolute,
            store_uri,
            DRUG_ENTITIES.model,
            revision,
            config,
        )
        logger.info(f'seeded {seeded} drug entity records')

    studies = read_aact_tables(source['aact'], ['studies'], config.work_path)['studies']
    extraction_dir = source['extraction'].rstrip('/')
    extractions = read_extraction(extraction_dir, config)

    run_id = new_run_id(s.snapshot)
    extraction_tag = EXTRACTION.release_tag()
    staged = {
        'stop_reasons': destination['stop_reasons'],
        'drug_entities': f'{extraction_dir}/{extraction_tag}/drug_entities',
    }
    stats = stage_trial_models(
        studies,
        extractions,
        models_uri,
        staged,
        store_uri,
        run_id,
        config,
        ner_batch_size=s.ner_batch_size,
        ner_model_batch_size=s.ner_model_batch_size,
        stop_reasons_batch_size=s.stop_reasons_batch_size,
        max_attempts=s.max_attempts,
        extraction_tag=extraction_tag,
    )
    logger.info(f'trial models staged: {stats}')
