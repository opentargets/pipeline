"""Implementation of the :py:mod:`pts.tasks.trial_models` task."""

from __future__ import annotations

import io
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import polars as pl
from loguru import logger
from otter.storage.synchronous.handle import StorageHandle
from otter.util.errors import NotFoundError

from pts.record_store import RECORD_SCHEMA, map_texts, publish_snapshot, text_sha256, write_run
from pts.transformers.clinical.helpers import sanitise_text
from pts.transformers.clinical.hf_models import offline_hf_models, revision_tags, staged_model_revisions
from pts.transformers.clinical.ner import DRUG_NER_MODEL, drug_entities_with_status, drug_ner_revision
from pts.transformers.clinical.providers.aact import llm_drug_labels
from pts.transformers.clinical.stop_reasons import (
    STOP_REASONS_MODEL,
    stop_reasons_revision,
    stop_reasons_with_status,
)

if TYPE_CHECKING:
    from collections.abc import Callable

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
    try:
        StorageHandle(f'{drug_store}/runs/{SEED_RUN_ID}/manifest.json', config=config).read_text()
    except NotFoundError:
        pass
    else:
        logger.info(f'{SEED_RUN_ID} already imported into {drug_store}')
        return 0

    data, _ = StorageHandle(ner_cache_uri, config=config).read()
    cache = (
        pl
        .read_parquet(io.BytesIO(data))
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


def tracked(compute: Callable[[list[str]], Any], calls: list[int]) -> Callable[[list[str]], Any]:
    """Wrap ``compute`` so the caller can tell whether it was asked for anything."""

    def wrapped(texts: list[str]) -> Any:
        calls.append(len(texts))
        return compute(texts)

    return wrapped


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
    stop_texts = why_stopped_texts(studies)
    drug_texts = llm_drug_labels(extractions)
    stop_calls: list[int] = []
    ner_calls: list[int] = []
    stop_revision = stop_reasons_revision(revisions[STOP_REASONS_MODEL])
    ner_revision = drug_ner_revision(revisions)

    with offline_hf_models(models_uri):
        stop = map_texts(
            stop_texts,
            tracked(lambda texts: stop_reasons_with_status(texts, stop_reasons_batch_size), stop_calls),
            f'{store_uri}/stop_reasons',
            STOP_REASONS_MODEL,
            stop_revision,
            run_id,
            config,
            max_attempts,
        )
        ner = map_texts(
            drug_texts,
            tracked(lambda texts: drug_entities_with_status(texts, ner_batch_size), ner_calls),
            f'{store_uri}/drug_entities',
            DRUG_NER_MODEL,
            ner_revision,
            run_id,
            config,
            max_attempts,
        )

    tags = revision_tags(revisions)
    stop_published = publish_snapshot(
        stop.rename({'output': 'categories'}).select('text', 'text_sha256', 'categories'),
        f'{destination["stop_reasons"]}/{tags["stop_reasons"]}',
        {'model': STOP_REASONS_MODEL, 'model_revision': stop_revision, 'model_revisions': revisions},
        config,
    )
    ner_published = publish_snapshot(
        ner.select(label=pl.col('text'), extracted_drugs=pl.col('output')),
        f'{destination["drug_entities"]}/{tags["drug_entities"]}',
        {'model': DRUG_NER_MODEL, 'model_revision': ner_revision, 'model_revisions': revisions},
        config,
    )
    return {
        'stop_reasons': {'texts': stop.height, 'computed': bool(stop_calls), 'published': stop_published},
        'drug_entities': {'texts': ner.height, 'computed': bool(ner_calls), 'published': ner_published},
    }
