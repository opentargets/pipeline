"""Transformer that stages the mechanisms of action the trial texts state for each drug.

The trials whose extraction names one drug are pooled by drug name. Each pool is a record in the store, keyed
by the content of its trials, so a pool whose trials did not change is not computed again. A changed pool whose
new text has no mechanism wording carries its previous output forward. Every pool's output, computed or not, is
checked against the current trial texts before it is published as a write-once snapshot beneath the extraction
it was computed from.
"""

from __future__ import annotations

import io
import os
from typing import TYPE_CHECKING, Any

import polars as pl
from loguru import logger
from otter.storage.synchronous.handle import StorageHandle
from otter.util.errors import NotFoundError
from pydantic import BaseModel, ConfigDict

from pts.transformers.clinical_mira.moa import pooling, quotes, schemas, staging
from pts.transformers.clinical_mira.record_store import (
    ComputeAborted,
    map_texts,
    new_run_id,
    publish_snapshot,
    read_records,
    write_run,
)
from pts.transformers.clinical_mira.staged_outputs import EXTRACTION, MOA
from pts.transformers.clinical_mira.staging_errors import ComputeTooLarge, SeedTooSmall

if TYPE_CHECKING:
    from collections.abc import Callable

    from otter.config.model import Config

SNAPSHOT_COLUMNS = [
    'drugName',
    'nctIds',
    'targetFromSource',
    'targetSymbolSuggestion',
    'actsOnReceptor',
    'targetType',
    'actionType',
    'mechanismOfAction',
    'relation',
    'statementType',
    'confidence',
    'effect_wording',
    'evidence',
    'isHumanTarget',
]
"""Columns of the published mechanisms."""


class AactMoaSettings(BaseModel):
    """Settings for the aact_moa transformer."""

    model_config = ConfigDict(extra='forbid')

    store_uri: str
    """Root of the record stores, an absolute URI. Sits outside the AACT version directory so a pool seen in an
        earlier version is not computed again."""
    snapshot: str
    """The AACT version. Prefixes the id of the runs this step writes and names the batch state file."""
    shard_size: int = 2000
    """Pools per shard. Each shard is written to the store as it finishes."""
    max_attempts: int = 3
    """Attempts per pool, within a run and across runs, before it is left out."""
    import_experiment_cache: str | None = None
    """The experiment's jsonl result cache to import into the store, once, before computing."""
    import_for_tag: str | None = None
    """The MOA tag the cache was computed for, or None to not import it. Any other tag than the current one is
        an error."""
    import_for_snapshot: str | None = None
    """The AACT version the cache is imported for. The cache is imported, and its coverage checked, only when
        this is ``snapshot``."""
    min_seeded_share: float = 0.5
    """The lowest share of the current pools the imported cache must cover, or the run aborts before computing."""
    max_compute: int | None = None
    """The most pools the run may compute, or None for no limit. More aborts the run before any batch."""
    workspace_id_env: str = 'ANTHROPIC_GOOGLE_CLOUD_WORKSPACE_ID'
    """Name of the environment variable holding the Claude workspace id."""


def _guarded(
    compute: Callable[[list[str]], list[tuple[str | None, str | None]]], workspace_id_env: str
) -> Callable[[list[str]], list[tuple[str | None, str | None]]]:
    """Wrap ``compute`` so it aborts the run when the workspace id variable is unset."""

    def guarded(keys: list[str]) -> list[tuple[str | None, str | None]]:
        if not os.environ.get(workspace_id_env):
            msg = f'{workspace_id_env} is not set, so {len(keys)} pools cannot be sent to Claude'
            raise ComputeAborted(msg)
        return compute(keys)

    return guarded


def _write_parquet(df: pl.DataFrame, uri: str, config: Config) -> None:
    buffer = io.BytesIO()
    df.write_parquet(buffer, compression='zstd')
    StorageHandle(uri, config=config).write(buffer.getvalue())


def _exists(uri: str, config: Config) -> bool:
    try:
        StorageHandle(uri, config=config).stat()
    except NotFoundError:
        return False
    return True


def _mechanism_rows(mechanisms: pl.DataFrame, pools: pl.DataFrame, listed: pl.DataFrame) -> pl.DataFrame:
    """The kept mechanisms with their pool's drug name and every trial of the pool."""
    trial_ids = listed.select('content_key', nctIds=pl.col('trials').list.eval(pl.element().struct.field('nct_id')))
    return (
        mechanisms
        .filter(pl.col('kept'))
        .join(pools.select('pool_id', 'drug_name', 'content_key'), on='pool_id', maintain_order='left')
        .join(trial_ids, on='content_key', maintain_order='left')
        .rename({'drug_name': 'drugName'})
        .sort('drugName', 'mechanism_idx')
        .select(SNAPSHOT_COLUMNS)
    )


def _pool_table(pools: pl.DataFrame, status: pl.DataFrame, carried: set[str]) -> pl.DataFrame:
    """One row per pool: its size, whether it was extracted, what its texts state and whether it was carried."""
    return (
        pools
        .select('pool_id', 'drug_name', 'n_trials', 'n_trials_used')
        .join(status, on='pool_id', maintain_order='left')
        .select(
            'drug_name',
            'n_trials',
            'n_trials_used',
            'extracted',
            status=pl
            .when(~pl.col('extracted'))
            .then(pl.lit('failed'))
            .when(pl.col('not_stated'))
            .then(pl.lit('not_stated'))
            .otherwise(pl.lit('stated')),
            carried=pl.col('pool_id').is_in(sorted(carried)),
        )
    )


def stage_moa(
    archive: str,
    extraction_dir: str,
    destination: str,
    settings: AactMoaSettings,
    config: Config,
    client: Any = None,
) -> dict[str, Any]:
    """Stage the mechanisms of action of the drugs of an AACT dump and publish them as a snapshot.

    Args:
        archive: The AACT dump archive. The tables required are restored inline.
        extraction_dir: The directory holding the extraction snapshot of the current tag.
        destination: The directory holding one directory per extraction tag, which receives the snapshot.
        settings: The validated settings.
        config: The otter config.
        client: The Anthropic client, ``AnthropicGoogleCloud()`` when ``None``.

    Returns:
        dict[str, Any]: The number of pools, seeded, carried and computed records, failed pools, mechanisms
        kept and dropped by the quote check, and whether a snapshot was published.

    Raises:
        ValueError: If ``import_for_tag`` is set and is not the current tag.
        SeedTooSmall: If the experiment cache is imported for the current tag and ``snapshot`` and it, or its
            import run, covers less than ``min_seeded_share`` of the pools.
        ComputeTooLarge: If more pools than ``max_compute`` have no ``ok`` record at the current tag after
            carry-forward.
        ComputeAborted: If there are pools to compute, no client is given and the workspace id variable is unset.
    """
    tag = MOA.tag()
    if settings.import_for_tag is not None and settings.import_for_tag != tag:
        msg = f'import_for_tag {settings.import_for_tag} is not the current tag {tag}; update or unset it'
        raise ValueError(msg)
    system_prompt = schemas.load_prompt('extract')
    extraction_tag = EXTRACTION.release_tag()
    extracted = EXTRACTION.read(f'{extraction_dir.rstrip("/")}/{extraction_tag}')

    texts, counts = pooling.match_texts(pooling.trial_texts(archive, str(config.work_path)), extracted)
    trials = pooling.single_drug_trials(extracted)
    pools = pooling.build_pools(trials, texts, char_budget=MOA.params['char_budget'])
    listed = staging.pool_trials(pools, texts, trials)
    logger.info(f'texts {counts}, {trials.height} single-drug trials, {pools.height} pools')

    store_uri = f'{settings.store_uri.rstrip("/")}/{MOA.name}'
    run_id = new_run_id(settings.snapshot)

    seeded = 0
    importing = (
        settings.import_experiment_cache is not None
        and settings.import_for_tag == tag
        and settings.import_for_snapshot == settings.snapshot
    )
    if settings.import_experiment_cache is not None and not importing:
        logger.info(
            f'not importing {settings.import_experiment_cache}: for tag {settings.import_for_tag} and snapshot '
            f'{settings.import_for_snapshot}, not {tag} and {settings.snapshot}'
        )
    if importing:
        seeded = staging.import_experiment_cache(
            settings.import_experiment_cache,
            pools,
            listed,
            system_prompt,
            store_uri,
            config,
            min_share=settings.min_seeded_share,
        )
        logger.info(f'imported {seeded} cached pools into {store_uri}')

    records = read_records(store_uri, config, pl.String())
    if importing:
        covered = (
            records
            .filter(pl.col('run_id') == staging.import_run_id())
            .filter(pl.col('text').is_in(pools['content_key'].implode()))['text']
            .n_unique()
        )
        if covered < settings.min_seeded_share * pools.height:
            msg = (
                f'{staging.import_run_id()} covers {covered} of {pools.height} pools, '
                f'below the minimum share {settings.min_seeded_share}'
            )
            raise SeedTooSmall(msg)

    done = set(records.filter((pl.col('status') == 'ok') & (pl.col('model_revision') == tag))['text'].to_list())
    carried = 0
    if MOA.params['carry_forward']:
        carried_records = staging.carry_forward(listed, texts, records, tag, f'{run_id}-carry')
        if carried_records.height:
            write_run(carried_records, store_uri, f'{run_id}-carry', config, pl.String())
        carried = carried_records.height
        done |= set(carried_records['text'].to_list())

    if settings.max_compute is not None:
        pending = len(set(pools['content_key'].to_list()) - done)
        if pending > settings.max_compute:
            msg = f'{pending} pools to compute, more than max_compute {settings.max_compute}'
            raise ComputeTooLarge(msg)

    compute = staging.moa_compute(
        pools,
        listed,
        system_prompt,
        f'{store_uri}/batches/{settings.snapshot}-{tag}.json',
        config.work_path,
        client=client,
    )
    mapped = map_texts(
        pools['content_key'],
        compute if client is not None else _guarded(compute, settings.workspace_id_env),
        store_uri,
        MOA.name,
        tag,
        run_id,
        config,
        max_attempts=settings.max_attempts,
        output_dtype=pl.String(),
        allow_failures=True,
        shard_size=settings.shard_size,
    )

    outputs = {key: staging.parse_output(output) for key, output in mapped.frame.select('text', 'output').iter_rows()}
    extractions = {}
    carried_pools = set()
    for pool_id, key in pools.select('pool_id', 'content_key').iter_rows():
        if key in outputs:
            extractions[pool_id] = schemas.MoaExtraction.model_validate(outputs[key]['extraction'])
            if outputs[key]['carried_from']:
                carried_pools.add(pool_id)

    mechanisms = quotes.check_mechanisms(extractions, pools, texts)
    status = quotes.pool_status(pools, extractions, mechanisms)
    rows = _mechanism_rows(mechanisms, pools, listed)

    uri = f'{destination.rstrip("/")}/{extraction_tag}/moa/{tag}'
    published = publish_snapshot(rows, uri, MOA.metadata(extraction_tag), config)
    if published or not _exists(f'{uri}/pools.parquet', config):
        _write_parquet(_pool_table(pools, status, carried_pools), f'{uri}/pools.parquet', config)

    return {
        'pools': pools.height,
        'seeded': seeded,
        'carried': carried,
        'computed': mapped.computed,
        'failed': mapped.failed,
        'mechanisms': rows.height,
        'dropped_by_quote_check': mechanisms.filter(~pl.col('kept')).height,
        'published': published,
    }


def aact_moa(source: dict[str, str], destination: dict[str, str], settings: dict[str, Any], config: Config) -> None:
    """Stage the mechanisms of action stated in AACT trial texts, reusing earlier outputs.

    Args:
        source: Keys ``aact``, the AACT dump archive, and ``extraction``, the directory of tagged extractions.
        destination: Key ``moa``, the directory of tagged extractions the snapshot is published beneath.
        settings: See :py:class:`AactMoaSettings`.
        config: The otter config.
    """
    s = AactMoaSettings.model_validate(settings)
    stats = stage_moa(source['aact'], source['extraction'], destination['moa'], s, config)
    logger.info(f'mechanisms staged: {stats}')
