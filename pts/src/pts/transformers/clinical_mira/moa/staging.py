"""Mechanism records keyed by the content of a pool, and the import of the experiment's cache."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import polars as pl
from loguru import logger
from otter.storage.synchronous.handle import StorageHandle

from pts.transformers.clinical_mira.aact_trial_extraction import IMPORT_COMPUTED_AT
from pts.transformers.clinical_mira.moa import llm, pooling, schemas
from pts.transformers.clinical_mira.moa.claude import read_usage, run_batch_cached
from pts.transformers.clinical_mira.record_store import import_run, record_schema, text_sha256
from pts.transformers.clinical_mira.staged_outputs import MOA

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

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


def request_text(drug_name: str, pool_text: str) -> str:
    """The user message of a pool's extraction request."""
    return f'Drug: {drug_name}\n\n{pool_text}'


def prompt_sha256(prompt: str) -> str:
    """The sha256 hex of a system prompt."""
    return hashlib.sha256(prompt.encode('utf-8')).hexdigest()


def record_output(
    drug_name: str,
    trials: list[dict],
    extraction: dict,
    usage: dict | None,
    carried_from: str | None,
    prompt_sha: str | None,
) -> str:
    """The JSON a record stores as its ``output``, ``prompt_sha256`` the sha256 of the system prompt it came from."""
    return json.dumps({
        'drug_name': drug_name,
        'trials': trials,
        'extraction': extraction,
        'usage': usage,
        'carried_from': carried_from,
        'prompt_sha256': prompt_sha,
    })


def parse_output(output: str) -> dict[str, Any]:
    """The fields of a record's ``output``, ``prompt_sha256`` null when the output predates it."""
    return {'prompt_sha256': None, **json.loads(output)}


def _cited(extraction: dict) -> set[str]:
    """The entry ids an extraction's evidence cites."""
    return {
        evidence['nctId']
        for mechanism in extraction.get('mechanisms') or []
        for evidence in mechanism.get('evidence') or []
        if evidence.get('nctId')
    }


def carry_forward(
    trials: pl.DataFrame, texts: pl.DataFrame, records: pl.DataFrame, tag: str, run_id: str
) -> pl.DataFrame:
    """Carry a drug's latest output to its changed pool when the change has no mechanism wording.

    A pool is carried when it has no ``ok`` record at ``tag``, an ``ok`` record at ``tag`` exists
    for its drug name, every trial added or edited since the latest such record has text
    without :py:data:`pooling.MOA_VOCAB` terms, and every entry of that record that its
    evidence cites, trial or publication, is still one of the pool's entries.

    Args:
        trials: Output of :py:func:`pool_trials`.
        texts: Trial prompts, ``id`` and ``prompt``.
        records: The record store's records.
        tag: The tag of the records to carry from.
        run_id: The run the carried records are written under.

    Returns:
        pl.DataFrame: Records ready for ``write_run``, with the pool's current ``trials``,
        the previous ``extraction`` and ``prompt_sha256``, no ``usage`` and ``carried_from`` the
        previous content key.
    """
    ok = records.filter((pl.col('status') == 'ok') & (pl.col('model_revision') == tag)).sort(
        'computed_at', maintain_order=True
    )
    done = set(ok['text'].to_list())
    latest: dict[str, dict] = {}
    for record in ok.to_dicts():
        parsed = parse_output(record['output'])
        latest[parsed['drug_name']] = {**parsed, 'content_key': record['text']}

    flagged = texts.select('id', mentions=pooling.mentions_mechanism(pl.col('prompt')))
    mentions = dict(zip(flagged['id'].to_list(), flagged['mentions'].to_list(), strict=True))

    now = datetime.now(UTC)
    rows = []
    for pool in trials.to_dicts():
        previous = latest.get(pool['drug_name'])
        if pool['content_key'] in done or previous is None or pool['trials'] is None:
            continue
        known = {t['nct_id']: t['content_sha256'] for t in previous['trials']}
        changed = [t['nct_id'] for t in pool['trials'] if known.get(t['nct_id']) != t['content_sha256']]
        if any(mentions.get(nct_id, True) for nct_id in changed):
            continue
        dropped = known.keys() - {t['nct_id'] for t in pool['trials']}
        if dropped & _cited(previous['extraction']):
            continue
        rows.append({
            'text_sha256': text_sha256(pool['content_key']),
            'text': pool['content_key'],
            'model': MOA.name,
            'model_revision': tag,
            'output': record_output(
                pool['drug_name'],
                pool['trials'],
                previous['extraction'],
                None,
                previous['content_key'],
                previous['prompt_sha256'],
            ),
            'status': 'ok',
            'error': None,
            'computed_at': now,
            'run_id': run_id,
        })
    logger.info(f'{len(rows)} of {trials.height} pools carried forward')
    return pl.DataFrame(rows, schema=record_schema(pl.String()))


def _read_cache_rows(cache_uri: str, config: Config) -> dict[str, dict]:
    text, _ = StorageHandle(cache_uri, config=config).read_text()
    rows = {}
    for line in text.splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row['key']] = row
    return rows


def import_run_id() -> str:
    """The id of the run the experiment's cache is imported as."""
    return f'import-moa-experiment-{MOA.digest}'


def import_papers_run_id() -> str:
    """The id of the run the cache computed with the publications prompt is imported as."""
    return f'{import_run_id()}-papers'


def import_experiment_cache(
    cache_uri: str,
    pools: pl.DataFrame,
    trials: pl.DataFrame,
    system_prompt: str,
    store_uri: str,
    config: Config,
    run_id: str | None = None,
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
        run_id: The id of the import run, :py:func:`import_run_id` when ``None``.

    Returns:
        int: The number of records written, 0 if the import run already exists.
    """
    run_id = run_id or import_run_id()

    def records() -> pl.DataFrame:
        cached = _read_cache_rows(cache_uri, config)
        listed = {row['content_key']: row['trials'] for row in trials.to_dicts()}
        rows = []
        for pool in pools.to_dicts():
            request = request_text(pool['drug_name'], pool['text'])
            row = cached.get(llm.result_key(request, system_prompt, schemas.MoaExtraction, MOA.model))
            if row is None:
                continue
            output = record_output(
                pool['drug_name'],
                listed[pool['content_key']],
                row['response'],
                row.get('usage'),
                None,
                prompt_sha256(system_prompt),
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


def _sha8(text: str) -> str:
    return prompt_sha256(text)[:8]


def _state_for(state_uri: str, prompt: str) -> str:
    stem, dot, suffix = state_uri.rpartition('.')
    return f'{stem}-{_sha8(prompt)}{dot}{suffix}' if dot else f'{state_uri}-{_sha8(prompt)}'


def adopt_records(records: pl.DataFrame, keys: list[str], from_tag: str, tag: str, run_id: str) -> pl.DataFrame:
    """Copy the ``ok`` records at ``from_tag`` of the given content keys to ``tag``.

    Keys that already have an ``ok`` record at ``tag`` are skipped. When a key has several records at
    ``from_tag`` the latest is copied.

    Args:
        records: The record store's records.
        keys: The content keys to adopt.
        from_tag: The tag to copy from.
        tag: The tag to copy to.
        run_id: The run the copies are written under.

    Returns:
        pl.DataFrame: Records ready for ``write_run``.
    """
    done = records.filter((pl.col('status') == 'ok') & (pl.col('model_revision') == tag))['text'].to_list()
    adopted = (
        records
        .filter(
            (pl.col('status') == 'ok')
            & (pl.col('model_revision') == from_tag)
            & pl.col('text').is_in(keys)
            & ~pl.col('text').is_in(done)
        )
        .sort('computed_at', maintain_order=True)
        .unique('text', keep='last', maintain_order=True)
        .with_columns(
            model_revision=pl.lit(tag),
            computed_at=pl.lit(datetime.now(UTC), dtype=record_schema(pl.String())['computed_at']),
            run_id=pl.lit(run_id),
        )
    )
    logger.info(f'{adopted.height} of {len(keys)} pools adopted from {from_tag}')
    return adopted.select(record_schema(pl.String()).names())


def moa_compute(
    pools: pl.DataFrame,
    trials: pl.DataFrame,
    system_prompt: str,
    state_uri: str,
    scratch: Path,
    client: Any = None,
    **batch_kwargs: Any,
) -> Callable[[list[str]], list[tuple[str | None, str | None]]]:
    """Build the ``compute`` of :py:func:`map_texts` that extracts mechanisms with Claude under one prompt.

    Args:
        pools: Output of :py:func:`pooling.build_pools`.
        trials: Output of :py:func:`pool_trials`.
        system_prompt: The extraction prompt.
        state_uri: Where the submitted batch ids are recorded, a path or URI. The file is named by the
            prompt's sha8.
        scratch: Local directory for the result cache, named by the prompt's sha8.
        client: The Anthropic client, ``AnthropicGoogleCloud()`` when ``None``.
        **batch_kwargs: Passed to :py:func:`run_batch_cached`.

    Returns:
        A function from content keys to ``(output, error)`` in the order of the keys.
    """
    texts = {
        row['content_key']: request_text(row['drug_name'], row['text'])
        for row in pools.select('content_key', 'drug_name', 'text').iter_rows(named=True)
    }
    listed = {row['content_key']: row for row in trials.to_dicts()}
    cache_path = scratch / f'moa_cache-{_sha8(system_prompt)}.jsonl'
    prompt_sha = prompt_sha256(system_prompt)

    def compute(keys: list[str]) -> list[tuple[str | None, str | None]]:
        results, errors = run_batch_cached(
            [(key, texts[key]) for key in keys],
            system_prompt=system_prompt,
            response_model=schemas.MoaExtraction,
            model=MOA.model,
            effort=MOA.params['effort'],
            cache_path=cache_path,
            state_uri=_state_for(state_uri, system_prompt),
            client=client,
            **batch_kwargs,
        )
        usage = read_usage(cache_path)
        out: list[tuple[str | None, str | None]] = []
        for key in keys:
            if key not in results:
                out.append((None, errors.get(key, 'missing')))
                continue
            result_key = llm.result_key(texts[key], system_prompt, schemas.MoaExtraction, MOA.model)
            output = record_output(
                listed[key]['drug_name'],
                listed[key]['trials'],
                results[key].model_dump(mode='json', by_alias=True),
                usage.get(result_key),
                None,
                prompt_sha,
            )
            out.append((output, None))
        return out

    return compute
