"""Append-only store of per-text model output, and write-once snapshots of it.

Some model passes are expensive and their input texts rarely change: a stop-reason
classifier over free text, a named-entity pass over drug labels. This module keeps
every result in a store with one identity per record: the text, the model and the
model revision. A run only computes the texts the store does not already answer.

Layout under ``store_uri``, one store per model::

    runs/<run_id>/records.parquet     what the run computed, ok and failed alike
    runs/<run_id>/manifest.json       written last, so a run that died midway is invisible

Runs are never rewritten. The current state of a record is derived from every
manifested run: the newest ``ok`` record of its key, else the newest ``failed``
one. "Newest" is the latest ``computed_at``, ties broken by ``run_id`` descending.
A failed record is retried by later runs until ``max_attempts`` failures have
accumulated; a text with an ``ok`` record is never computed again.

A finished store is published as a snapshot under a version directory::

    data.parquet
    metadata.json                     written last

Snapshots are write-once: republishing identical content is a no-op, and
publishing different content to the same location is refused.
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

RECORD_SCHEMA = pl.Schema({
    'text_sha256': pl.String,
    'text': pl.String,
    'model': pl.String,
    'model_revision': pl.String,
    'output': pl.List(pl.String),
    'status': pl.String,
    'error': pl.String,
    'computed_at': pl.Datetime(time_unit='us', time_zone='UTC'),
    'run_id': pl.String,
})
"""Columns of every record. ``status`` is ``ok`` or ``failed``; ``output`` is null when failed."""

KEY = ['text_sha256', 'model', 'model_revision']
"""Columns that identify a record."""

MAX_EXAMPLES = 5
"""Texts quoted in the error raised for texts that have no ok record."""


class RecordStoreError(Exception):
    """A record store operation could not be completed."""


class SnapshotConflictError(Exception):
    """A snapshot already exists at the location with different content."""


def text_sha256(text: str) -> str:
    """Return the hex sha256 of the UTF-8 encoding of a text."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _read_parquet(uri: str, config: Config) -> pl.DataFrame:
    """Read a parquet file through otter storage, so it works on GCS and locally alike."""
    data, _ = StorageHandle(uri, config=config).read()
    return pl.read_parquet(io.BytesIO(data))


def _write_parquet(df: pl.DataFrame, uri: str, config: Config) -> None:
    """Write a parquet file through otter storage, so it works on GCS and locally alike."""
    buf = io.BytesIO()
    df.write_parquet(buf, compression='zstd')
    StorageHandle(uri, config=config).write(buf.getvalue())


def _write_json(document: dict[str, Any], uri: str, config: Config) -> None:
    StorageHandle(uri, config=config).write_text(json.dumps(document, indent=2, sort_keys=True))


def _exists(uri: str, config: Config) -> bool:
    try:
        StorageHandle(uri, config=config).read_text()
    except NotFoundError:
        return False
    return True


def read_records(store_uri: str, config: Config) -> pl.DataFrame:
    """Read every record of every finished run.

    Args:
        store_uri: Root of the store, an absolute URI.
        config: The otter config, used to resolve storage.

    Returns:
        pl.DataFrame: The records of the runs whose manifest exists, or an empty
            frame with ``RECORD_SCHEMA`` when there are none.
    """
    try:
        manifests = sorted(StorageHandle(f'{store_uri}/runs', config=config).glob('*/manifest.json'))
    except NotFoundError:
        manifests = []

    frames = [_read_parquet(m.removesuffix('manifest.json') + 'records.parquet', config) for m in manifests]
    if not frames:
        return pl.DataFrame(schema=RECORD_SCHEMA)
    return pl.concat(frames).select(RECORD_SCHEMA.names()).cast(RECORD_SCHEMA)


def write_run(records: pl.DataFrame, store_uri: str, run_id: str, config: Config) -> None:
    """Write one run: its records first, then the manifest that makes it visible.

    Args:
        records: The records of the run, with the columns of ``RECORD_SCHEMA``.
        store_uri: Root of the store, an absolute URI.
        run_id: Identifies the run.
        config: The otter config, used to resolve storage.

    Raises:
        RecordStoreError: If the run already has a manifest.
    """
    run_uri = f'{store_uri}/runs/{run_id}'
    if _exists(f'{run_uri}/manifest.json', config):
        msg = f'run {run_id} already exists in {store_uri}'
        raise RecordStoreError(msg)

    records = records.select(RECORD_SCHEMA.names()).cast(RECORD_SCHEMA)
    _write_parquet(records, f'{run_uri}/records.parquet', config)

    models = records.select('model', 'model_revision').unique().sort('model', 'model_revision')
    manifest = {
        'run_id': run_id,
        'ok': records.filter(pl.col('status') == 'ok').height,
        'failed': records.filter(pl.col('status') == 'failed').height,
        'models': models.to_dicts(),
        'written_at': datetime.now(UTC).isoformat(),
    }
    _write_json(manifest, f'{run_uri}/manifest.json', config)
    logger.info(f'wrote run {run_id}: {manifest["ok"]} ok, {manifest["failed"]} failed')


def _current(records: pl.DataFrame) -> pl.DataFrame:
    """Reduce records to one per key: the newest ok, else the newest failed."""
    return (
        records
        .with_columns(is_ok=pl.col('status') == 'ok')
        .sort(['is_ok', 'computed_at', 'run_id'], descending=True)
        .unique(subset=KEY, keep='first', maintain_order=True)
        .drop('is_ok')
    )


def map_texts(
    texts: pl.Series,
    compute: Callable[[list[str]], list[tuple[list[str] | None, str | None]]],
    store_uri: str,
    model: str,
    model_revision: str,
    run_id: str,
    config: Config,
    max_attempts: int = 3,
) -> pl.DataFrame:
    """Return the model output for every distinct text, computing only what the store lacks.

    A text is computed when it has no record for this model and revision, or only
    failed ones numbering fewer than ``max_attempts``. All of them go to one
    ``compute`` call, with the texts distinct and sorted. If ``compute`` raises,
    every text of the call is recorded as failed with the exception text.

    Args:
        texts: The texts to map. Must not contain nulls.
        compute: Maps a list of texts to one ``(output, None)`` or ``(None, error)``
            per text, in the same order.
        store_uri: Root of the store, an absolute URI.
        model: Name of the model.
        model_revision: Revision of the model.
        run_id: Identifies the run this call writes, if it computes anything.
        config: The otter config, used to resolve storage.
        max_attempts: Failed records after which a text is no longer retried.

    Returns:
        pl.DataFrame: ``text``, ``text_sha256`` and ``output`` per distinct text, sorted by text.

    Raises:
        ValueError: If ``texts`` contains nulls.
        RecordStoreError: If any text still has no ok record once the run is written.
    """
    if texts.null_count():
        msg = f'texts contains {texts.null_count()} null values'
        raise ValueError(msg)

    wanted = (
        texts
        .cast(pl.String)
        .unique()
        .sort()
        .to_frame('text')
        .with_columns(text_sha256=pl.col('text').map_elements(text_sha256, return_dtype=pl.String))
    )

    existing = read_records(store_uri, config).filter(
        (pl.col('model') == model) & (pl.col('model_revision') == model_revision)
    )
    ok_hashes = existing.filter(pl.col('status') == 'ok').select('text_sha256').unique()
    attempts = (
        existing
        .filter(pl.col('status') == 'failed')
        .join(ok_hashes, on='text_sha256', how='anti')
        .group_by('text_sha256')
        .agg(attempts=pl.len())
    )
    todo = (
        wanted
        .join(ok_hashes, on='text_sha256', how='anti')
        .join(attempts, on='text_sha256', how='left')
        .filter(pl.col('attempts').fill_null(0) < max_attempts)
    )
    logger.info(f'{wanted.height} distinct texts, {todo.height} to compute')

    fresh = pl.DataFrame(schema=RECORD_SCHEMA)
    if todo.height:
        todo_texts = todo['text'].to_list()
        try:
            results = compute(todo_texts)
        except Exception as exc:
            logger.error(f'compute failed for {len(todo_texts)} texts: {exc}')
            results = [(None, str(exc))] * len(todo_texts)
        if len(results) != len(todo_texts):
            msg = f'compute returned {len(results)} results for {len(todo_texts)} texts'
            raise RecordStoreError(msg)

        fresh = pl.DataFrame(
            {
                'text_sha256': todo['text_sha256'],
                'text': todo['text'],
                'model': model,
                'model_revision': model_revision,
                'output': [r[0] for r in results],
                'status': ['failed' if r[0] is None else 'ok' for r in results],
                'error': [r[1] for r in results],
                'computed_at': datetime.now(UTC),
                'run_id': run_id,
            },
            schema=RECORD_SCHEMA,
        )
        write_run(fresh, store_uri, run_id, config)

    current = _current(pl.concat([existing, fresh]))
    done = wanted.join(current.filter(pl.col('status') == 'ok').select('text_sha256', 'output'), on='text_sha256')
    if done.height < wanted.height:
        missing = wanted.join(done.select('text_sha256'), on='text_sha256', how='anti')['text'].to_list()
        examples = ', '.join(repr(t) for t in missing[:MAX_EXAMPLES])
        msg = f'{len(missing)} texts have no ok record for {model}@{model_revision}, e.g. {examples}'
        raise RecordStoreError(msg)

    return done.select('text', 'text_sha256', 'output').sort('text')


def publish_snapshot(df: pl.DataFrame, uri: str, metadata: dict[str, Any], config: Config) -> bool:
    """Publish a snapshot, once.

    ``data.parquet`` is written first and ``metadata.json`` last, so a snapshot
    without metadata is an unfinished one.

    Args:
        df: The snapshot contents.
        uri: Directory of the snapshot, an absolute URI.
        metadata: Describes the snapshot. Must be JSON serialisable.
        config: The otter config, used to resolve storage.

    Returns:
        bool: True if the snapshot was written, False if an identical one was already there.

    Raises:
        SnapshotConflictError: If a snapshot with different data or metadata is already there.
    """
    wanted_metadata = json.loads(json.dumps(metadata))
    try:
        text, _ = StorageHandle(f'{uri}/metadata.json', config=config).read_text()
    except NotFoundError:
        _write_parquet(df, f'{uri}/data.parquet', config)
        _write_json(wanted_metadata, f'{uri}/metadata.json', config)
        logger.info(f'published snapshot {uri} with {df.height} rows')
        return True

    if json.loads(text) != wanted_metadata:
        msg = f'snapshot {uri} exists with different metadata'
        raise SnapshotConflictError(msg)

    published = _read_parquet(f'{uri}/data.parquet', config)
    if published.columns != df.columns or not published.sort(published.columns).equals(df.sort(df.columns)):
        msg = f'snapshot {uri} exists with different data'
        raise SnapshotConflictError(msg)

    logger.info(f'snapshot {uri} already published, nothing to do')
    return False
