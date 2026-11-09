"""Literature mechanism records keyed by a drug's selected sentences, and the import of the experiment's seed."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import polars as pl
from loguru import logger
from otter.storage.synchronous.handle import StorageHandle

from pts.transformers.clinical_mira.aact_trial_extraction import IMPORT_COMPUTED_AT
from pts.transformers.clinical_mira.moa import llm, schemas
from pts.transformers.clinical_mira.moa.claude import read_usage, run_batch_cached
from pts.transformers.clinical_mira.moa.staging import prompt_sha256
from pts.transformers.clinical_mira.record_store import import_run, record_schema, text_sha256
from pts.transformers.clinical_mira.staged_outputs import MOA_LITERATURE

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from otter.config.model import Config


def record_output(
    parent: str,
    sentences: list[dict],
    extraction: dict,
    usage: dict | None,
    carried_from: str | None,
    prompt_sha: str | None,
) -> str:
    """The JSON a record stores as its ``output``, ``prompt_sha256`` the sha256 of the system prompt it came from."""
    return json.dumps({
        'parent': parent,
        'sentences': sentences,
        'extraction': extraction,
        'usage': usage,
        'carried_from': carried_from,
        'prompt_sha256': prompt_sha,
    })


def parse_output(output: str) -> dict[str, Any]:
    """The fields of a record's ``output``."""
    return json.loads(output)


def _cited(extraction: dict) -> set[str]:
    """The document ids an extraction's evidence cites."""
    return {
        evidence['nctId']
        for mechanism in extraction.get('mechanisms') or []
        for evidence in mechanism.get('evidence') or []
        if evidence.get('nctId')
    }


def import_run_id() -> str:
    """The id of the run the experiment's seed is imported as."""
    return f'import-moa-literature-experiment-{MOA_LITERATURE.digest}'


def carry_forward(requests: pl.DataFrame, records: pl.DataFrame, tag: str, run_id: str) -> pl.DataFrame:
    """Carry a drug's latest output to its changed selection when no new sentence has a cue.

    A drug is carried when its ``content_key`` has no ``ok`` record at ``tag``, an ``ok`` record at
    ``tag`` outside the import run :py:func:`import_run_id` exists for its parent, no sentence of the
    current selection missing from the latest such record (by ``doc`` and ``sha256``) has ``cue``,
    and every document of that record that its evidence cites is still in the selection.

    Args:
        requests: Output of :py:func:`literature.requests`.
        records: The record store's records.
        tag: The tag of the records to carry from.
        run_id: The run the carried records are written under.

    Returns:
        pl.DataFrame: Records ready for ``write_run``, with the drug's current ``sentences``, the
        previous ``extraction`` and ``prompt_sha256``, no ``usage`` and ``carried_from`` the previous
        content key.
    """
    ok = records.filter((pl.col('status') == 'ok') & (pl.col('model_revision') == tag))
    done = set(ok['text'].to_list())
    latest: dict[str, dict] = {}
    for record in ok.filter(pl.col('run_id') != import_run_id()).sort('computed_at', maintain_order=True).to_dicts():
        parsed = parse_output(record['output'])
        latest[parsed['parent']] = {**parsed, 'content_key': record['text']}

    now = datetime.now(UTC)
    rows = []
    for drug in requests.select('parent', 'content_key', 'sentences').to_dicts():
        previous = latest.get(drug['parent'])
        if drug['content_key'] in done or previous is None:
            continue
        known = {(s['doc'], s['sha256']) for s in previous['sentences']}
        if any(s['cue'] for s in drug['sentences'] if (s['doc'], s['sha256']) not in known):
            continue
        dropped = {doc for doc, _ in known} - {s['doc'] for s in drug['sentences']}
        if dropped & _cited(previous['extraction']):
            continue
        rows.append({
            'text_sha256': text_sha256(drug['content_key']),
            'text': drug['content_key'],
            'model': MOA_LITERATURE.name,
            'model_revision': tag,
            'output': record_output(
                drug['parent'],
                drug['sentences'],
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
    logger.info(f'{len(rows)} of {requests.height} drugs carried forward')
    return pl.DataFrame(rows, schema=record_schema(pl.String()))


def _read_seed_rows(seed_uri: str, config: Config) -> list[dict]:
    text, _ = StorageHandle(seed_uri, config=config).read_text()
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def import_experiment_cache(
    cache_uri: str,
    requests: pl.DataFrame,
    system_prompt: str,
    store_uri: str,
    config: Config,
) -> int:
    """Import the experiment's seed into the record store, once.

    The import is the run :py:func:`import_run_id`. A seed row becomes an ``ok`` record whose text
    is its ``content_key`` when that key equals the content key of the current request for its
    ``parent`` and its ``prompt_sha256`` is the sha256 of ``system_prompt``; other rows are skipped.

    Args:
        cache_uri: The seed, a jsonl of ``content_key``, ``parent``, ``response``, ``usage``,
            ``prompt_sha256`` and ``run``; a path or URI.
        requests: Output of :py:func:`literature.requests`.
        system_prompt: The extraction prompt of the current requests.
        store_uri: Root of the store, an absolute URI.
        config: The otter config, used to resolve storage.

    Returns:
        int: The number of records written, 0 if the import run already exists.
    """
    run_id = import_run_id()
    prompt_sha = prompt_sha256(system_prompt)

    def records() -> pl.DataFrame:
        seed = _read_seed_rows(cache_uri, config)
        current = {row['parent']: row for row in requests.select('parent', 'content_key', 'sentences').to_dicts()}
        rows = []
        other_prompt = 0
        for row in seed:
            if row.get('prompt_sha256') != prompt_sha:
                other_prompt += 1
                continue
            drug = current.get(row['parent'])
            if drug is None or drug['content_key'] != row['content_key']:
                continue
            rows.append({
                'text_sha256': text_sha256(drug['content_key']),
                'text': drug['content_key'],
                'model': MOA_LITERATURE.name,
                'model_revision': MOA_LITERATURE.tag(),
                'output': record_output(
                    drug['parent'], drug['sentences'], row['response'], row.get('usage'), None, prompt_sha
                ),
                'status': 'ok',
                'error': None,
                'computed_at': IMPORT_COMPUTED_AT,
                'run_id': run_id,
            })
        logger.info(
            f'{len(rows)} of {len(seed)} seed rows match a current request in {cache_uri}; '
            f'{other_prompt} skipped for another prompt'
        )
        return pl.DataFrame(rows, schema=record_schema(pl.String()))

    return import_run(records, store_uri, run_id, config, output_dtype=pl.String())


def literature_compute(
    requests: pl.DataFrame,
    system_prompt: str,
    state_uri: str,
    scratch: Path,
    client: Any = None,
    **batch_kwargs: Any,
) -> Callable[[list[str]], list[tuple[str | None, str | None]]]:
    """Build the ``compute`` of :py:func:`map_texts` that extracts mechanisms from sentences with Claude.

    Args:
        requests: Output of :py:func:`literature.requests`.
        system_prompt: The extraction prompt.
        state_uri: Where the submitted batch ids are recorded, a path or URI.
        scratch: Local directory for the result cache.
        client: The Anthropic client, ``AnthropicGoogleCloud()`` when ``None``.
        **batch_kwargs: Passed to :py:func:`run_batch_cached`.

    Returns:
        A function from content keys to ``(output, error)`` in the order of the keys.
    """
    listed = {
        row['content_key']: row for row in requests.select('parent', 'content_key', 'request', 'sentences').to_dicts()
    }
    cache_path = scratch / 'moa_literature_cache.jsonl'
    prompt_sha = prompt_sha256(system_prompt)

    def compute(keys: list[str]) -> list[tuple[str | None, str | None]]:
        results, errors = run_batch_cached(
            [(key, listed[key]['request']) for key in keys],
            system_prompt=system_prompt,
            response_model=schemas.MoaExtraction,
            model=MOA_LITERATURE.model,
            effort=MOA_LITERATURE.params['effort'],
            cache_path=cache_path,
            state_uri=state_uri,
            client=client,
            **batch_kwargs,
        )
        usage = read_usage(cache_path)
        out: list[tuple[str | None, str | None]] = []
        for key in keys:
            if key not in results:
                out.append((None, errors.get(key, 'missing')))
                continue
            drug = listed[key]
            result_key = llm.result_key(drug['request'], system_prompt, schemas.MoaExtraction, MOA_LITERATURE.model)
            output = record_output(
                drug['parent'],
                drug['sentences'],
                results[key].model_dump(mode='json', by_alias=True),
                usage.get(result_key),
                None,
                prompt_sha,
            )
            out.append((output, None))
        return out

    return compute
