"""Content-addressed cache for expensive per-record work.

Some enrichments cost real money or real time for every record they touch: an
LLM call per clinical trial, a model inference pass per free-text field. Running
them again for every release is waste, because between one release and the next
almost nothing changes.

This module keeps their results in a cache keyed by a hash of the exact input
that produced them, so a run only pays for what is genuinely new.

Keying on the input rather than on the record id is the point. Trial text gets
revised upstream, and prompts and schemas change on our side. A key derived
from those turns every such change into a cache miss, which is what stops a
stale result being served forever after a prompt edit — the failure worth
designing against, because nothing about it looks wrong. Operational choices
such as the model and system instructions are deliberately not cache identity.

Layout under ``cache_uri``::

    latest.json                       pointer, rewritten last so a crash is invisible
    snapshots/<timestamp>/rows.parquet
    staging/<run_id>/shard-0000.parquet

The cache lives outside any release or dataset version directory. That is
deliberate: a record whose input did not change between two versions has to be
a hit, and it cannot be one if the cache is filed under the version.
"""

from __future__ import annotations

import hashlib
import io
import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import polars as pl
from loguru import logger
from otter.storage.synchronous.handle import StorageHandle
from otter.util.errors import NotFoundError

if TYPE_CHECKING:
    from collections.abc import Callable

    from otter.config.model import Config

KEY_SEPARATOR = '\x1f'
"""Separator between key parts. A control character, so it cannot occur inside one."""

METADATA_SCHEMA: dict[str, Any] = {
    'cache_key': pl.String,
    'computed_at': pl.Datetime(time_unit='us', time_zone='UTC'),
}
"""Columns every cache row carries, on top of whatever the compute function returns."""


def cache_key(*parts: str) -> str:
    """Return a stable hash of everything that influenced a result.

    Pass the values that define the result's identity, in a fixed order. For
    LLM extraction this is the rendered trial prompt and output schema. The
    model and system instructions are intentionally excluded by the caller:
    changing either must not invalidate existing accepted results.

    Args:
        *parts: The values to hash, in a fixed order.

    Returns:
        str: A hex sha256 digest.
    """
    return hashlib.sha256(KEY_SEPARATOR.join(parts).encode('utf-8')).hexdigest()


def _read_parquet(uri: str, config: Config) -> pl.DataFrame:
    """Read a parquet file through otter storage, so it works on GCS and locally alike."""
    data, _ = StorageHandle(uri, config=config).read()
    return pl.read_parquet(io.BytesIO(data))


def _write_parquet(df: pl.DataFrame, uri: str, config: Config) -> None:
    """Write a parquet file through otter storage, so it works on GCS and locally alike."""
    buf = io.BytesIO()
    df.write_parquet(buf, compression='zstd')
    StorageHandle(uri, config=config).write(buf.getvalue())


def read_cache(cache_uri: str, config: Config) -> pl.DataFrame:
    """Read the current cache snapshot.

    Args:
        cache_uri: Root of the cache, an absolute URI.
        config: The otter config, used to resolve storage.

    Returns:
        pl.DataFrame: The cached rows, or an empty frame with only the metadata
            columns when the cache does not exist yet.
    """
    empty = pl.DataFrame(schema=METADATA_SCHEMA)

    try:
        pointer, _ = StorageHandle(f'{cache_uri}/latest.json', config=config).read_text()
    except NotFoundError:
        logger.info(f'no cache at {cache_uri}, starting from empty')
        return empty

    snapshot = json.loads(pointer)['snapshot']
    rows = _read_parquet(f'{cache_uri}/snapshots/{snapshot}/rows.parquet', config=config)
    logger.info(f'loaded {rows.height} cached rows from snapshot {snapshot}')
    return rows


def write_cache(rows: pl.DataFrame, cache_uri: str, config: Config, timestamp: str) -> str:
    """Write a new cache snapshot and repoint ``latest.json`` at it.

    The pointer is written last, so a run that dies midway leaves the previous
    snapshot in place and is invisible to the next one.

    Args:
        rows: The full cache contents to write.
        cache_uri: Root of the cache, an absolute URI.
        config: The otter config, used to resolve storage.
        timestamp: Snapshot name. Passed in rather than generated here so the
            caller can tie it to the run.

    Returns:
        str: The name of the snapshot written.
    """
    _write_parquet(rows, f'{cache_uri}/snapshots/{timestamp}/rows.parquet', config=config)
    pointer = json.dumps({'snapshot': timestamp, 'rows': rows.height}, indent=2)
    StorageHandle(f'{cache_uri}/latest.json', config=config).write_text(pointer)
    logger.info(f'wrote cache snapshot {timestamp} with {rows.height} rows')
    return timestamp


def _read_shards(staging_uri: str, config: Config) -> pl.DataFrame:
    """Read whatever shards a previous attempt of this run already finished."""
    try:
        paths = sorted(StorageHandle(staging_uri, config=config).glob('shard-*.parquet'))
    except NotFoundError:
        return pl.DataFrame()

    if not paths:
        return pl.DataFrame()

    shards = pl.concat([_read_parquet(p, config=config) for p in paths], how='diagonal_relaxed')
    logger.info(f'resuming: {shards.height} rows recovered from {len(paths)} staged shards')
    return shards


def cached_map(
    records: pl.DataFrame,
    compute: Callable[[pl.DataFrame], pl.DataFrame],
    cache_uri: str,
    config: Config,
    run_id: str,
    timestamp: str,
    shard_size: int = 2000,
) -> pl.DataFrame:
    """Compute a result for every record, reusing the cache for the ones already done.

    Only results are cached. A record ``compute`` failed on simply has no row,
    which makes it indistinguishable from one that was never attempted — so the
    next run tries it again. That is deliberate: this pipeline runs a few times
    a year, and re-attempting a few hundred stubborn records on each run costs
    far less than the bookkeeping needed to remember not to.

    Work is sharded, and each shard is staged to storage as soon as it finishes.
    A retry of the task picks the finished shards back up rather than paying for
    them twice, which matters mostly on a first full run.

    Args:
        records: The records to map. Must carry a ``cache_key`` column, plus
            whatever ``compute`` reads.
        compute: Maps the subset of ``records`` that missed the cache to a frame
            of results, carrying a ``cache_key`` column. Records it fails on
            should be left out of the returned frame.
        cache_uri: Root of the cache, an absolute URI.
        config: The otter config, used to resolve storage.
        run_id: Identifies the staging area for this run. Reuse it across
            retries of the same run to resume; change it to start clean.
        timestamp: Name for the snapshot this run writes.
        shard_size: Records per shard.

    Returns:
        pl.DataFrame: One row per record in ``records`` that has a result,
            cached or freshly computed. Records still missing are absent, so the
            caller can see the coverage gap rather than having it papered over.
    """
    wanted = records.select('cache_key').unique()
    cached = read_cache(cache_uri, config)

    staging_uri = f'{cache_uri}/staging/{run_id}'
    staged = _read_shards(staging_uri, config)

    done = cached.select('cache_key')
    if not staged.is_empty():
        done = pl.concat([done, staged.select('cache_key')])

    todo = records.join(done.unique(), on='cache_key', how='anti')
    known = wanted.height - todo.height
    logger.info(f'{wanted.height} records requested, {known} already known, {todo.height} to compute')

    fresh = [staged] if not staged.is_empty() else []
    for shard_no, offset in enumerate(range(0, todo.height, shard_size)):
        shard = todo.slice(offset, shard_size)
        logger.info(f'computing shard {shard_no} ({shard.height} records)')

        result = _stamp(compute(shard))
        _write_parquet(result, f'{staging_uri}/shard-{shard_no:04d}.parquet', config=config)
        fresh.append(result)

    combined = pl.concat([cached, *fresh], how='diagonal_relaxed') if fresh else cached
    # newest row per key wins, so a record recomputed after an earlier failure
    # replaces nothing stale
    combined = combined.sort('computed_at', descending=True).unique(subset='cache_key', keep='first')

    write_cache(combined, cache_uri, config, timestamp)
    return combined.join(wanted, on='cache_key', how='semi')


def _stamp(result: pl.DataFrame) -> pl.DataFrame:
    """Add the metadata columns a cache row needs, leaving any the caller already set."""
    if result.is_empty():
        return pl.DataFrame(schema=METADATA_SCHEMA)

    if 'computed_at' in result.columns:
        return result
    return result.with_columns(computed_at=pl.lit(datetime.now(UTC)).cast(METADATA_SCHEMA['computed_at']))
