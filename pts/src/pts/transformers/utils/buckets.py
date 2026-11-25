"""Hash buckets of a large dataset on local disk, so a step holds one bucket at a time instead of the whole input.

`write_buckets` reads the input part by part and writes each part's rows to `<directory>/<bucket>/<part>.parquet`
by `hash(key) % bucket_count`; `read_bucket` reads one bucket back; `for_each_bucket` runs a function over every
bucket, `concurrency` at a time.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TypeVar

import polars as pl

T = TypeVar('T')


def bucket_of(key: str, bucket_count: int) -> pl.Expr:
    """The bucket of each row: `hash(key) % bucket_count`, with a fixed seed."""
    return (pl.col(key).hash(seed=0) % bucket_count).alias('bucket')


def _map(function: Callable[[int], T], items: Sequence[int], concurrency: int) -> list[T]:
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(function, item) for item in items]
        try:
            return [future.result() for future in futures]
        except BaseException:
            pool.shutdown(cancel_futures=True)
            raise


def write_buckets(
    parts: Sequence[str],
    prepare: Callable[[pl.DataFrame], pl.DataFrame],
    key: str,
    bucket_count: int,
    directory: Path,
    concurrency: int,
) -> int:
    """Split every part of a dataset into buckets under `directory`.

    Args:
        parts: the parquet files of the dataset.
        prepare: applied to each part as read; its result is what is bucketed, and must have `key`.
        key: the column whose hash picks the bucket.
        bucket_count: the number of buckets.
        directory: an empty local directory.
        concurrency: parts processed at a time.

    Returns:
        The number of rows written.
    """

    def split(index: int) -> int:
        frame = prepare(pl.read_parquet(parts[index]))
        for (bucket,), rows in frame.with_columns(bucket_of(key, bucket_count)).partition_by(
            'bucket', as_dict=True, include_key=False
        ).items():
            folder = directory / f'{bucket:05d}'
            folder.mkdir(exist_ok=True)
            rows.write_parquet(folder / f'{index:05d}.parquet', compression='lz4')
        return frame.height

    return sum(_map(split, range(len(parts)), concurrency))


def read_bucket(directory: Path, bucket: int) -> pl.DataFrame | None:
    """The rows of one bucket, or None for a bucket no part reached."""
    folder = directory / f'{bucket:05d}'
    if not folder.is_dir():
        return None
    return pl.read_parquet(sorted(folder.glob('*.parquet')))


def for_each_bucket(function: Callable[[int], T], bucket_count: int, concurrency: int) -> list[T]:
    """`function(bucket)` for every bucket, `concurrency` at a time; the first failure cancels the rest."""
    return _map(function, range(bucket_count), concurrency)
