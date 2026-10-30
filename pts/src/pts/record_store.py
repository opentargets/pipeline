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


def record_schema(output_dtype: pl.DataType) -> pl.Schema:
    """Return the columns of every record of a store whose model output has ``output_dtype``.

    ``status`` is ``ok`` or ``failed``; ``output`` is null when failed.
    """
    return pl.Schema({
        'text_sha256': pl.String,
        'text': pl.String,
        'model': pl.String,
        'model_revision': pl.String,
        'output': output_dtype,
        'status': pl.String,
        'error': pl.String,
        'computed_at': pl.Datetime(time_unit='us', time_zone='UTC'),
        'run_id': pl.String,
    })


LIST_OUTPUT = pl.List(pl.String)
"""Output dtype of the stores that hold a list of labels per text."""

RECORD_SCHEMA = record_schema(LIST_OUTPUT)
"""Columns of every record of a store that holds a list of strings per text."""

KEY = ['text_sha256', 'model', 'model_revision']
"""Columns that identify a record."""

MAX_EXAMPLES = 5
"""Texts quoted in the error raised for texts that have no ok record."""


class RecordStoreError(Exception):
    """A record store operation could not be completed."""


class ComputeAborted(Exception):  # noqa: N818
    """Raised by a compute to stop the run without recording any text of the call as failed.

    ``finished`` holds the records of the rounds that completed before it, if any.
    """

    finished: pl.DataFrame | None = None


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


def run_exists(store_uri: str, run_id: str, config: Config) -> bool:
    """Return whether the store has a finished run ``run_id``, that is, one whose manifest exists."""
    return _exists(f'{store_uri}/runs/{run_id}/manifest.json', config)


def read_records(store_uri: str, config: Config, output_dtype: pl.DataType = LIST_OUTPUT) -> pl.DataFrame:
    """Read every record of every finished run.

    Args:
        store_uri: Root of the store, an absolute URI.
        config: The otter config, used to resolve storage.
        output_dtype: Dtype of the store's ``output`` column.

    Returns:
        pl.DataFrame: The records of the runs whose manifest exists, or an empty
            frame with the store's record schema when there are none.
    """
    try:
        manifests = sorted(StorageHandle(f'{store_uri}/runs', config=config).glob('*/manifest.json'))
    except NotFoundError:
        manifests = []

    schema = record_schema(output_dtype)
    frames = [_read_parquet(m.removesuffix('manifest.json') + 'records.parquet', config) for m in manifests]
    if not frames:
        return pl.DataFrame(schema=schema)
    return pl.concat(frames).select(schema.names()).cast(schema)


def write_run(
    records: pl.DataFrame,
    store_uri: str,
    run_id: str,
    config: Config,
    output_dtype: pl.DataType = LIST_OUTPUT,
) -> None:
    """Write one run: its records first, then the manifest that makes it visible.

    Args:
        records: The records of the run, with the columns of the store's record schema.
        store_uri: Root of the store, an absolute URI.
        run_id: Identifies the run.
        config: The otter config, used to resolve storage.
        output_dtype: Dtype of the store's ``output`` column.

    Raises:
        RecordStoreError: If the run already has a manifest.
    """
    run_uri = f'{store_uri}/runs/{run_id}'
    if run_exists(store_uri, run_id, config):
        msg = f'run {run_id} already exists in {store_uri}'
        raise RecordStoreError(msg)

    schema = record_schema(output_dtype)
    records = records.select(schema.names()).cast(schema)
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


def _compute_round(
    todo: pl.DataFrame,
    compute: Callable[[list[str]], list[tuple[Any, str | None]]],
    model: str,
    model_revision: str,
    run_id: str,
    schema: pl.Schema,
) -> pl.DataFrame:
    """Compute one round over ``todo`` and return its records."""
    todo_texts = todo['text'].to_list()
    try:
        results = compute(todo_texts)
    except ComputeAborted:
        raise
    except Exception as exc:
        logger.error(f'compute failed for {len(todo_texts)} texts: {exc}')
        msg = f'compute raised for a call of {len(todo_texts)} texts: {exc}'
        raise ComputeAborted(msg) from exc
    if len(results) != len(todo_texts):
        msg = f'compute returned {len(results)} results for {len(todo_texts)} texts'
        raise RecordStoreError(msg)

    return pl.DataFrame(
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
        schema=schema,
    )


def _compute_with_retries(
    todo: pl.DataFrame,
    compute: Callable[[list[str]], list[tuple[Any, str | None]]],
    model: str,
    model_revision: str,
    run_id: str,
    schema: pl.Schema,
    max_attempts: int,
) -> pl.DataFrame:
    """Compute ``todo``, recomputing failures, one call per round, until each text reaches ``max_attempts``.

    Raises:
        ComputeAborted: If ``compute`` raises it, carrying the records of the finished rounds.
    """
    rounds = []
    pending = todo
    while pending.height:
        try:
            fresh = _compute_round(pending, compute, model, model_revision, run_id, schema)
        except ComputeAborted as exc:
            if rounds:
                exc.finished = pl.concat(rounds)
            raise
        rounds.append(fresh)
        failed = fresh.filter(pl.col('status') == 'failed').select('text_sha256')
        pending = (
            pending
            .join(failed, on='text_sha256')
            .with_columns(attempts=pl.col('attempts') + 1)
            .filter(pl.col('attempts') < max_attempts)
        )
    return pl.concat(rounds)


def tracked(compute: Callable[[list[str]], Any], calls: list[int]) -> Callable[[list[str]], Any]:
    """Wrap ``compute`` so the caller can tell whether it was asked for anything."""

    def wrapped(texts: list[str]) -> Any:
        calls.append(len(texts))
        return compute(texts)

    return wrapped


def map_texts(
    texts: pl.Series,
    compute: Callable[[list[str]], list[tuple[Any, str | None]]],
    store_uri: str,
    model: str,
    model_revision: str,
    run_id: str,
    config: Config,
    max_attempts: int = 3,
    output_dtype: pl.DataType = LIST_OUTPUT,
    allow_failures: bool = False,
    shard_size: int | None = None,
) -> pl.DataFrame:
    """Return the model output for every distinct text, computing only what the store lacks.

    A text is computed when it has no record for this model and revision, or only
    failed ones numbering fewer than ``max_attempts``. All of them go to one
    ``compute`` call, with the texts distinct and sorted. An exception from
    ``compute`` aborts the call, as :py:class:`ComputeAborted` chained to it: the
    call records nothing, so no text uses up an attempt. A text that fails alone
    is returned as ``(None, error)``. An abort in a retry round still writes the
    rounds before it.

    With ``allow_failures`` the texts that fail are recomputed in the same run,
    one ``compute`` call per round, until their failures, earlier runs' included,
    reach ``max_attempts``. Those still failing are left out of the result with a
    warning.

    With ``shard_size`` the texts to compute are split, in sorted order, into
    shards of that size. Each is computed and written as the run
    ``<run_id>-<n:04d>`` before the next starts, so a crash keeps the finished ones.

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
        output_dtype: Dtype of the store's ``output`` column.
        allow_failures: Whether texts that keep failing are left out instead of raising.
        shard_size: Number of texts per run, or None for a single run.

    Returns:
        pl.DataFrame: ``text``, ``text_sha256`` and ``output`` per distinct text, sorted by text.

    Raises:
        ValueError: If ``texts`` contains nulls, or ``shard_size`` is not positive.
        RecordStoreError: If any text still has no ok record once the run is written
            and ``allow_failures`` is False.
        ComputeAborted: If ``compute`` raises. Shards written before stay written, and
            so do the finished rounds of the shard it aborts.
    """
    if texts.null_count():
        msg = f'texts contains {texts.null_count()} null values'
        raise ValueError(msg)
    if shard_size is not None and shard_size < 1:
        msg = f'shard_size must be positive, got {shard_size}'
        raise ValueError(msg)

    schema = record_schema(output_dtype)
    wanted = (
        texts
        .cast(pl.String)
        .unique()
        .sort()
        .to_frame('text')
        .with_columns(text_sha256=pl.col('text').map_elements(text_sha256, return_dtype=pl.String))
    )

    existing = read_records(store_uri, config, output_dtype).filter(
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
        .with_columns(pl.col('attempts').fill_null(0))
        .filter(pl.col('attempts') < max_attempts)
    )
    logger.info(f'{wanted.height} distinct texts, {todo.height} to compute')

    shards = [todo] if shard_size is None else [todo.slice(i, shard_size) for i in range(0, todo.height, shard_size)]
    written = [pl.DataFrame(schema=schema)]
    for n, shard in enumerate(shards):
        if not shard.height:
            continue
        shard_run_id = run_id if shard_size is None else f'{run_id}-{n:04d}'
        if allow_failures:
            try:
                fresh = _compute_with_retries(shard, compute, model, model_revision, shard_run_id, schema, max_attempts)
            except ComputeAborted as exc:
                if exc.finished is not None:
                    write_run(exc.finished, store_uri, shard_run_id, config, output_dtype)
                raise
        else:
            fresh = _compute_round(shard, compute, model, model_revision, shard_run_id, schema)
        write_run(fresh, store_uri, shard_run_id, config, output_dtype)
        written.append(fresh)

    current = _current(pl.concat([existing, *written]))
    done = wanted.join(current.filter(pl.col('status') == 'ok').select('text_sha256', 'output'), on='text_sha256')
    if done.height < wanted.height:
        missing = wanted.join(done.select('text_sha256'), on='text_sha256', how='anti')['text'].to_list()
        examples = ', '.join(repr(t) for t in missing[:MAX_EXAMPLES])
        if not allow_failures:
            msg = f'{len(missing)} texts have no ok record for {model}@{model_revision}, e.g. {examples}'
            raise RecordStoreError(msg)
        logger.warning(f'{len(missing)} texts have no ok record for {model}@{model_revision}, e.g. {examples}')

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
    uri = uri.rstrip('/')
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
