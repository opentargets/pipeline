"""Flat association timeseries view: one row per association, aggregation value and year."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import polars as pl
from loguru import logger
from ot_datamodels.polars import to_polars_schema
from ot_datamodels.views.association_timeseries import AssociationTimeseries
from otter.config.model import Config

from pts.schemas.dataset.conform import conform
from pts.task_outputs import writes
from pts.transformers.utils.dataset import check_destinations_absent, scan_dataset

DEFAULT_FILE_COUNT = 300
DEFAULT_CONCURRENCY = 8
ROW_GROUP_SIZE = 500_000
INPUTS = {
    'association_by_datasource_indirect': False,
    'association_overall_indirect': False,
    'association_by_datasource_direct': True,
    'association_overall_direct': True,
}
SORT = ['partitionKey', 'aggregationValue', 'year']
SPLIT_COLUMNS = ['diseaseId', 'targetId', 'aggregationType', 'aggregationValue', 'timeseries']
TIMESERIES = pl.List(
    pl.Struct({
        'year': pl.Int32(),
        'associationScore': pl.Float64(),
        'novelty': pl.Float64(),
        'yearlyEvidenceCount': pl.Int64(),
    })
)


@writes(AssociationTimeseries)
def association_timeseries_view(
    source: dict[str, str],
    destination: str,
    settings: dict[str, Any],
    config: Config | None,
) -> None:
    """Explode the four association datasets into a timeseries view of `file_count` sorted files.

    Each input is read once and split in memory by targetId hash; each output file is built from
    its own split only, so every partitionKey lands in one file.

    Args:
        source: paths keyed by the four association datasets.
        destination: output directory.
        settings: optional file_count and concurrency, the number of files built at a time.
        config: unused.
    """
    file_count = int(settings.get('file_count', DEFAULT_FILE_COUNT))
    concurrency = int(settings.get('concurrency', DEFAULT_CONCURRENCY))
    check_destinations_absent([destination])
    if '://' not in destination:
        os.makedirs(destination)

    pieces: dict[int, list[pl.DataFrame]] = {index: [] for index in range(file_count)}
    for name, is_direct in INPUTS.items():
        logger.info(f'splitting {name}')
        frame = (
            scan_dataset(source[name])
            .select(
                *SPLIT_COLUMNS,
                pl.lit(is_direct).alias('isDirect'),
                (pl.col('targetId').hash(seed=0) % file_count).cast(pl.UInt32).alias('file'),
            )
            .with_columns(
                pl.col('timeseries').cast(TIMESERIES), pl.col('aggregationType', 'aggregationValue').cast(pl.String)
            )
            .collect()
        )
        for (index,), piece in frame.partition_by('file', as_dict=True, include_key=False).items():
            pieces[index].append(piece)
        del frame

    def write(index: int) -> None:
        split = pieces.pop(index)
        view = (
            (pl.concat(split) if split else pl.DataFrame(schema=_split_schema()))
            .lazy()
            .explode('timeseries')
            .drop_nulls('timeseries')
            .unnest('timeseries')
            .with_columns(
                pl.concat_str(
                    pl.col('targetId'),
                    pl.col('diseaseId'),
                    pl.col('isDirect').cast(pl.String),
                    separator='_',
                    ignore_nulls=True,
                ).alias('partitionKey')
            )
            .select(to_polars_schema(AssociationTimeseries).names())
            .sort(SORT)
        )
        conform(view, AssociationTimeseries).collect().write_parquet(
            f'{destination}/part-{index:05d}.parquet',
            compression='zstd',
            statistics=True,
            row_group_size=ROW_GROUP_SIZE,
        )
        logger.info(f'association_timeseries_view file {index + 1}/{file_count} written')

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(write, index) for index in range(file_count)]
        try:
            for future in futures:
                future.result()
        except BaseException:
            pool.shutdown(cancel_futures=True)
            raise


def _split_schema() -> pl.Schema:
    return pl.Schema({
        'diseaseId': pl.String(),
        'targetId': pl.String(),
        'aggregationType': pl.String(),
        'aggregationValue': pl.String(),
        'timeseries': TIMESERIES,
        'isDirect': pl.Boolean(),
    })
