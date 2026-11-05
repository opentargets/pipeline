"""Mechanism records keyed by the content of a pool, and the import of the experiment's cache."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import polars as pl
from loguru import logger
from otter.storage.synchronous.handle import StorageHandle

from pts.transformers.clinical_mira.aact_trial_extraction import IMPORT_COMPUTED_AT
from pts.transformers.clinical_mira.moa import llm, pooling, schemas
from pts.transformers.clinical_mira.record_store import import_run, record_schema, text_sha256
from pts.transformers.clinical_mira.staged_outputs import MOA

if TYPE_CHECKING:
    from otter.config.model import Config

TRIAL = pl.Struct({'nct_id': pl.String, 'content_sha256': pl.String})


def pool_trials(pools: pl.DataFrame, texts: pl.DataFrame, trials: pl.DataFrame) -> pl.DataFrame:
    """List every trial of each pool with the hash of its content, within the budget or not.

    Args:
        pools: Output of :py:func:`pooling.build_pools`.
        texts: Trial prompts, ``id`` and ``prompt``.
        trials: The trials the pools were built from, ``nct_id`` and ``drug_name``.

    Returns:
        pl.DataFrame: ``content_key``, ``drug_name`` and ``trials``, the pool's trials sorted by ``nct_id``.
    """
    hashed = (
        trials
        .join(texts.select(pl.col('id').alias('nct_id'), 'prompt'), on='nct_id', how='inner', maintain_order='left')
        .select(
            'drug_name',
            'nct_id',
            content_sha256=pl.col('prompt').map_elements(pooling.trial_content_sha256, return_dtype=pl.String),
        )
        .sort('drug_name', 'nct_id')
        .group_by('drug_name', maintain_order=True)
        .agg(trials=pl.struct('nct_id', 'content_sha256'))
    )
    return pools.select('content_key', 'drug_name').join(hashed, on='drug_name', how='left', maintain_order='left')


def record_output(
    drug_name: str, trials: list[dict], extraction: dict, usage: dict | None, carried_from: str | None
) -> str:
    """The JSON a record stores as its ``output``."""
    return json.dumps({
        'drug_name': drug_name,
        'trials': trials,
        'extraction': extraction,
        'usage': usage,
        'carried_from': carried_from,
    })


def parse_output(output: str) -> dict[str, Any]:
    """The fields of a record's ``output``."""
    return json.loads(output)


def _read_cache_rows(cache_uri: str, config: Config) -> dict[str, dict]:
    text, _ = StorageHandle(cache_uri, config=config).read_text()
    rows = {}
    for line in text.splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row['key']] = row
    return rows


def import_experiment_cache(
    cache_uri: str,
    pools: pl.DataFrame,
    trials: pl.DataFrame,
    system_prompt: str,
    store_uri: str,
    config: Config,
) -> int:
    """Import the experiment's result cache into the record store, once.

    The import is the run ``import-moa-experiment-<digest>``. A cached result becomes an
    ``ok`` record whose text is the content key of the pool it was computed for, matched by
    the experiment's cache key of the pool text, ``system_prompt`` and the extraction schema.

    Args:
        cache_uri: The experiment's jsonl cache, a path or URI.
        pools: Output of :py:func:`pooling.build_pools`.
        trials: Output of :py:func:`pool_trials`.
        system_prompt: The prompt the cache was computed with.
        store_uri: Root of the store, an absolute URI.
        config: The otter config, used to resolve storage.

    Returns:
        int: The number of records written, 0 if the import run already exists.
    """
    run_id = f'import-moa-experiment-{MOA.digest}'

    def records() -> pl.DataFrame:
        cached = _read_cache_rows(cache_uri, config)
        listed = {row['content_key']: row['trials'] for row in trials.to_dicts()}
        rows = []
        for pool in pools.to_dicts():
            row = cached.get(llm.result_key(pool['text'], system_prompt, schemas.MoaExtraction, MOA.model))
            if row is None:
                continue
            output = record_output(
                pool['drug_name'], listed[pool['content_key']], row['response'], row.get('usage'), None
            )
            rows.append({
                'text_sha256': text_sha256(pool['content_key']),
                'text': pool['content_key'],
                'model': MOA.name,
                'model_revision': MOA.tag(),
                'output': output,
                'status': 'ok',
                'error': None,
                'computed_at': IMPORT_COMPUTED_AT,
                'run_id': run_id,
            })
        logger.info(f'{len(rows)} of {pools.height} pools found in {cache_uri}')
        return pl.DataFrame(rows, schema=record_schema(pl.String()))

    return import_run(records, store_uri, run_id, config, output_dtype=pl.String())
